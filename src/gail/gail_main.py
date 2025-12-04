"""
Link: https://github.com/HumanCompatibleAI/imitation/blob/master/docs/tutorials/3_train_gail.ipynb
"""

import torch
import os
import random
import argparse
import numpy as np
import pandas as pd
from multiprocessing import get_context
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.networks import RunningNorm

from gail_utils import OneHotRewardNet, OneHotWrapper, RedditEnv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=10_000, help="GAIL total timesteps")
    p.add_argument("--workers", type=int, default=60)
    p.add_argument("--use-running-norm", dest="use_running_norm",
               action=argparse.BooleanOptionalAction, default=True,
               help="Use running normalization (default: on)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

args = parse_args()

# CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Global knobs (read by workers)
WORKERS = args.workers
NODES = args.nodes

# Basic seeding (single-seed run)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def gail_process(traj: np.ndarray, tp: np.ndarray):
    n_states, n_actions, _ = tp.shape

    # Create the base environment.
    base_env = RedditEnv(n_states, n_actions, tp)
    wrapped_env = OneHotWrapper(base_env, n_states)
    env = DummyVecEnv([lambda: wrapped_env])

    # Build expert trajectories
    expert_trajs = []
    for traj_i in traj:
        traj_i = np.array(traj_i)
        obs = traj_i[:, 0].astype(np.int64)
        acts = traj_i[:, 1]
        onehot_obs = np.zeros((len(obs), n_states), dtype=np.float32)
        onehot_obs[np.arange(len(obs)), obs] = 1.0
        onehot_next_obs = np.concatenate([onehot_obs[1:], onehot_obs[-1:]], axis=0)
        dones = np.zeros(len(obs), dtype=bool); dones[-1] = True
        expert_trajs.append({
            "obs": onehot_obs,
            "acts": acts,
            "next_obs": onehot_next_obs,
            "dones": dones
        })

    # ---- Generator (policy) with PPO on CPU ----
    gen_algo = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        ent_coef=0.01,
        gamma=args.gamma,
        verbose=False,
        policy_kwargs=dict(net_arch=[dict(pi=[16,16], vf=[16,16])]),
        device="cpu",
        seed=args.seed,
    )

    # Custom one-hot reward net
    reward_net = OneHotRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=(RunningNorm if args.use_running_norm else None),
        nodes=NODES,
    )

    # ---- GAIL trainer ----
    gail_trainer = GAIL(
        demonstrations=expert_trajs,
        venv=env,
        demo_batch_size=len(expert_trajs[0]["obs"]),
        gen_algo=gen_algo,
        init_tensorboard=False,
        n_disc_updates_per_round=5,
        reward_net=reward_net,
    )

    gail_trainer.train(args.steps)

    # Extract final policy mapping
    policy_map = np.zeros((n_states, n_actions), dtype=np.float32)
    for s in range(n_states):
        obs = np.zeros(n_states, dtype=np.float32); obs[s] = 1.0
        obs = obs.reshape(1, -1)
        obs_tensor = torch.tensor(obs).float().to(gen_algo.policy.device)
        with torch.no_grad():
            dist_wrapper = gen_algo.policy.get_distribution(obs_tensor)
            action_probs = dist_wrapper.distribution.probs.cpu().numpy().flatten()
        policy_map[s] = action_probs

    return policy_map

def _worker(args):
    """Unpack a (traj, tp) tuple for Pool.imap."""
    return gail_process(*args)

if __name__ == "__main__":
    # Explicitly use spawn on macOS
    ctx = get_context("spawn")

    root = "/raoscratch/home/phschnei/projects/io-detection-reddit/"
    # Load your data
    filepath = root + "data-analysis/sampled_matched_perturbed_df_compat.pkl"
    df = pd.read_pickle(filepath)

    # Build argument list
    args_list = [
        (row["traj_perturbed"], row["tp_perturbed_legalised"])
        for _, row in df.iterrows()
    ]

    # Prepare output list
    policies = [None] * len(args_list)

    # Parallel map with a tqdm progress bar
    with ctx.Pool(processes=WORKERS) as pool:
        for idx, policy in enumerate(tqdm(
                pool.imap(_worker, args_list),
                total=len(args_list),
                desc="GAIL jobs",
        )):
            policies[idx] = policy

    # Store back into DataFrame
    df["policy"] = policies

    root = "/raoscratch/home/phschnei/projects/io-detection-reddit/"
    # Load your data
    filepath = root + "data-analysis/sampled_matched_perturbed_df_w_gail_opt.pkl"
    df.to_pickle(filepath)
