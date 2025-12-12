import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from src.irl.max_ent_irl import RedditIRL, DeepMaximumEntropy


# Note: this script was run on a HPC cluster


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
                        