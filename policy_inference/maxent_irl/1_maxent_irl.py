import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from joblib import Parallel, delayed


# Note: this script was run on a HPC cluster


class RedditIRL:
    def __init__(self, n_actions, n_states, dynamics):
        # for reference dynamics = tp
        self.dynamics = dynamics
        self.n_states = n_states
        self.n_actions = n_actions

class DeepMaximumEntropy:
    def __init__(self, env, trajectories, features,
                 layers=(3, 3), lr=0.01, discount=0.9, l1=0.0, l2=0.0):
        self.env = env
        self.trajectories = torch.LongTensor(trajectories)
        self.features = torch.FloatTensor(features)
        self.discount = discount
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self._eps = 1e-6

        # dynamics: [S × A × S']
        self.dynamics = torch.FloatTensor(env.dynamics)

        # build network
        modules = []
        last = features.shape[1]
        for h in layers:
            modules += [nn.Linear(last, h), nn.Sigmoid()]
            last = h
        self.net = nn.Sequential(*modules)
        self.alpha = nn.Linear(last, 1, bias=False)

        # init weights
        torch.manual_seed(12345)
        nn.init.normal_(self.alpha.weight, mean=0.0, std=1.0)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                nn.init.normal_(m.bias, mean=0.0, std=1.0)

        # collect parameters and accumulators
        self.param_list = list(self.net.parameters()) + list(self.alpha.parameters())
        self.hist = [torch.zeros_like(p.data) for p in self.param_list]

    def forward(self, features):
        phi = self.net(features)
        r = self.alpha(phi).view(-1)
        return (r - r.mean()) / (r.std() + 1e-8)
    
    def state_visitation_frequency(self):
        """
        Empirical, *discounted* state‐visitation frequency:
        \hatμ_E[s] = (∑_{i=1}^N ∑_{t=0}^{L_i−1} γ^t · 1_{s_t^{(i)}=s})
                    / (∑_{i=1}^N ∑_{t=0}^{L_i−1} γ^t).
        """
        device = self.dynamics.device
        svf = torch.zeros(self.env.n_states, device=device)
        denom = 0.0

        for traj in self.trajectories:
            for t, (s, _) in enumerate(traj):
                w = self.discount ** t
                svf[s] += w
                denom += w

        return svf / max(denom, 1e-8)
    
    def state_visitation_frequency_vec(self):
        """
        Vectorized, discounted empirical visitation frequency:

        μ_E[s] = 
            ( ∑_{i=1}^N ∑_{t=0}^{L−1} γ^t · 1_{s_t^{(i)} = s} )
            / ( ∑_{i=1}^N ∑_{t=0}^{L−1} γ^t ).

        Assumes:
        — self.trajectories is a LongTensor of shape [N, L, 2],
            where each entry is (state, action) and all trajectories have the same length L.
        — self.discount is the γ discount factor.
        — self.env.n_states = S.

        Returns:
        A FloatTensor of shape [S] on the same device as `self.dynamics`, 
        containing the discounted visitation frequencies μ_E.
        """
        device = self.dynamics.device

        # 1) Extract all state indices: shape [N, L]
        #    (we ignore actions here, so take index 0 of the last dim)
        states = self.trajectories[..., 0]             # torch.LongTensor [N × L]

        N, L = states.shape
        S   = self.env.n_states

        # 2) Precompute discount powers [γ^0, γ^1, …, γ^(L−1)] on `device`
        gamma_pows = (self.discount ** torch.arange(L, device=device, dtype=torch.float32))
        #    gamma_pows: FloatTensor [L]

        # 3) Build a [N, L] weight matrix by broadcasting
        #    (each of the N trajectories reuses the same row of [γ^0 … γ^(L−1)])
        weights = gamma_pows.unsqueeze(0)               # FloatTensor [1 × L]
        #    After broadcasting, it behaves like [N × L].

        # 4) Flatten both `states` and `weights` into vectors of length N·L
        flat_states  = states.reshape(-1)               # LongTensor  [N*L]
        flat_weights = weights.expand(N, L).reshape(-1) # FloatTensor [N*L]

        # 5) Scatter‐add into an [S] accumulator
        svf = torch.zeros(S, device=device)             # FloatTensor [S]
        svf.scatter_add_(0, flat_states, flat_weights)
        # Now svf[s] = sum of γ^t over all (i,t) such that states[i,t] == s.

        # 6) Denominator = sum of all weights = ∑_{i=1}^N ∑_{t=0}^{L−1} γ^t
        denom = flat_weights.sum().clamp(min=1e-8)

        return svf / denom

    def expected_svf(self, policy):
        """
        Expected, *discounted* state‐visitation frequency under π:
        \hatμ_π[s] = (∑_{i=1}^N ∑_{t=0}^{L_i−1} γ^t · Pr(s_t = s | π))
                    / (∑_{i=1}^N ∑_{t=0}^{L_i−1} γ^t).
        """
        device = self.dynamics.device

        # 1) Empirical initial-state distribution (per trajectory)
        prob0 = torch.zeros(self.env.n_states, device=device)
        for traj in self.trajectories:
            s0 = int(traj[0, 0])
            prob0[s0] += 1.0
        prob0 /= len(self.trajectories)

        # 2) One-step transition matrix under policy π
        #    policy: [S × A], dynamics: [S × A × S'], result x: [S × S']
        x = (policy[:, :, None] * self.dynamics).sum(dim=1)  # [S × S']

        # 3) Propagate and accumulate with discount
        exp_counts = torch.zeros(self.env.n_states, device=device)
        denom = 0.0

        for traj in self.trajectories:
            # t = 0
            mu = prob0.clone()
            w0 = self.discount ** 0  # = 1.0
            exp_counts += mu * w0
            denom += w0

            # t = 1 … L_i−1
            for t in range(1, traj.shape[0]):
                mu = mu @ x
                wt = self.discount ** t
                exp_counts += mu * wt
                denom += wt

        return exp_counts / max(denom, 1e-8)
    
    @torch.no_grad()
    def expected_svf_vec(self, policy):
        """
        Same interface as above, but now each trajectory is padded to [L_max,2]
        with sentinel state = -1. We derive lengths by counting until state < 0.
        """
        device = self.dynamics.device
        S = self.env.n_states

        # --- 1) Empirical initial-state distribution prob0[s] = (# trajectories starting at s)/N ---
        start_states = self.trajectories[:, 0, 0]                 # [N]
        prob0 = torch.zeros(S, device=device)
        prob0.scatter_add_(0,
                        start_states,
                        torch.ones(start_states.size(0), device=device))
        prob0 /= self.trajectories.size(0)

        # --- 2) One-step transition matrix under policy π---
        x = (policy[:, :, None] * self.dynamics).sum(dim=1)        # [S × S']

        # --- 3) Infer lengths from sentinel state = -1 ---
        # Extract only the state column, shape [N, L_max]:
        states = self.trajectories[:, :, 0]                       # [N, L_max]
        # valid_mask[i,t] = 1 if states[i,t] >= 0 (i.e. a real state), else 0
        valid_mask = (states >= 0).float()                        # [N, L_max]
        # lengths[i] = number of valid steps in trajectory i
        lengths = valid_mask.sum(dim=1).long()                    # [N]

        # --- 4) Build mask[i,t] = 1 if t < lengths[i], else 0 (same result) ---
        N, L_max, _ = self.trajectories.shape
        idx = torch.arange(L_max, device=device).unsqueeze(0)      # [1, L_max]
        lengths_unsq = lengths.unsqueeze(1)                       # [N,1]
        mask = (idx < lengths_unsq).float()                        # [N, L_max]
        # M_t = how many traj survive to time t
        M_t = mask.sum(dim=0)                                      # [L_max]

        # --- 5) Precompute discount powers γ^t ---
        gamma_pows = (self.discount ** torch.arange(
                        L_max, device=device, dtype=torch.float32))  # [L_max]

        # --- 6) Build weights w_t = γ^t * M_t ---
        w_t = gamma_pows * M_t                                    # [L_max]

        # --- 7) Iterate to compute μ_t and accumulate ---
        exp_counts = torch.zeros(S, device=device)                # [S]
        mu = prob0.clone()                                          # μ_0

        for t in range(L_max):
            exp_counts += w_t[t] * mu
            mu = mu @ x

        # --- 8) Normalize by Σ_t w_t ---
        denom = w_t.sum().clamp(min=1e-8)
        return exp_counts / denom

    @torch.no_grad()
    def value_iteration(self, rewards, threshold=1e-2):
        device = self.dynamics.device
        r = rewards.to(device)
        V = torch.zeros(self.env.n_states, device=device)
        for _ in range(1000):
            V_prev = V
            # Q[s,a] = sum_{s'} P(s'|s,a) * [r[s'] + discount * V[s']]
            Q = torch.matmul(self.dynamics, (r + self.discount * V))
            V = Q.max(dim=1)[0]
            if (V - V_prev).abs().max() < threshold:
                break
        Q = torch.matmul(self.dynamics, (r + self.discount * V))
        Q = Q - Q.max(dim=1, keepdim=True)[0]
        return torch.softmax(Q, dim=1)

    def train(self, n_epochs, save_rewards=True, verbose=False):
        self.rews = []
        
        # Compute empirical discounted visitation frequency using vectorized version
        svf = self.state_visitation_frequency_vec().to(self.dynamics.device)
        
        for e in range(n_epochs):
            rewards = self.forward(self.features.to(self.dynamics.device))
            if verbose:
                print(f"Epoch {e:3d}, rewards = {rewards.detach().cpu().numpy()}")
            if save_rewards:
                self.rews.append(rewards.detach().cpu().numpy())
            policy = self.value_iteration(rewards)
            
            # Compute expected visitation frequency under current policy (vectorized)
            exp_svf = self.expected_svf_vec(policy)
            
            # Backpropagate
            for p in self.param_list:
                if p.grad is not None:
                    p.grad.zero_()
            # Loss = (svf - exp_svf) · rewards
            loss = (svf - exp_svf).dot(rewards)
            loss.backward()
            for idx, p in enumerate(self.param_list):
                g = p.grad.data   # ∂J/∂p, shape = p.shape

                if idx < len(self.param_list) - 1:
                    # L1 gradient:  λ₁ * sign(p)
                    # L2 gradient:  2 * λ₂ * p
                    penalty_grad = self.l1 * torch.sign(p.data) \
                                + 2.0 * self.l2 * p.data
                    w_grad = g - penalty_grad
                else:
                    # No regularization on the alpha layer
                    w_grad = g   

                # AdaGrad accumulator
                self.hist[idx] += w_grad.pow(2)

                # Gradient-ascent step
                p.data.add_(
                    (w_grad / (self.hist[idx].sqrt() + self._eps))
                    * self.lr
                )
        return self.forward(self.features.to(self.dynamics.device)).detach()


# IRL parameters
discount = .95
epochs = 1500
learning_rate = 0.05
l1 = 0
l2 = 0.5
structure = (3, 3)

num_s = 12
num_a = 6

feature_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
                        
def irl(row):
    reddit_irl = RedditIRL(num_a, num_s, row.tp_n_legalised)
    dme = DeepMaximumEntropy(
        reddit_irl, row.traj_n,
        feature_matrix, structure,
        learning_rate, discount, l1=l1, l2=l2
    )
    r = dme.train(epochs).squeeze()
    policy = dme.value_iteration(r)

    return {
        "discount":discount,
        "epochs":epochs,
        "learning_rate":learning_rate,
        "l1":l1,
        "l2":l2,
        "user": row.user,
        "run": row.run,
        "n": row.n,
        "reward": r,
        "policy": policy
    }                        
                        

# perturb
df = pd.read_pickle("sampled_matched_perturbed_df.pkl")
results = Parallel(n_jobs=-1)(
    delayed(irl)(row) for _, row in df.iterrows()
)
result_df = pd.DataFrame(results)
output_dir = f"perturb_irl/"
os.makedirs(output_dir, exist_ok=True)
result_df.to_pickle(os.path.join(output_dir, "sampled_matched_perturbed_df_irl.pkl"))


# first n
df = pd.read_pickle("first_n_df.pkl")
results = Parallel(n_jobs=-1)(
    delayed(irl)(row) for _, row in df.iterrows()
)
result_df = pd.DataFrame(results)
output_dir = f"first_n_irl/"
os.makedirs(output_dir, exist_ok=True)
result_df.to_pickle(os.path.join(output_dir, "first_n_irl.pkl"))
                        