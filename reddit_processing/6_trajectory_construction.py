import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
from tqdm.contrib.concurrent import process_map  
from functools import partial
import multiprocessing
import json
sys.path.append("../../src/irl")
import reddit_traj_construction


def process_user(user, load_dir, output_dir, start_date, end_date):
    results = []
    user_path = os.path.join(load_dir, user)
    if os.path.exists(user_path) and os.path.isdir(user_path):
        if not os.path.isfile(output_dir + "/" + user + ".npz"):
            print(user)
            tp, traj_counts, trajectories, feature_matrix = reddit_traj_construction.construct_tp_traj(
                user,
                load_dir,
                output_dir,
                start_date=start_date,
                end_date=end_date
            )

            if tp is not None:
                results.append((tp, traj_counts, trajectories))
    return results



# Use process_map for multiprocessing with progress bar
if __name__ == '__main__':
    
    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = int(config["collection_start_date"])
        collection_end_date = int(config["collection_end_date"])
        root_output_dir_path = config["root_output_dir_path"]
        deberta_weight_path = config["deberta_weight_path"]
        processed_per_user_path =  root_output_dir_path + "/4_per_user_processing" 

    # rus active timeframe 
    start_date = "2015-01-02"
    end_date = "2018-04-11"

    load_dir = processed_per_user_path + "/4_per_user_processing_collated"
    output_dir = root_output_dir_path + "/users_tp_traj"
    os.makedirs(output_dir, exist_ok=True)

    all_users = sorted(os.listdir(load_dir))
    print(len(all_users))

    # Partial to fix the non-user arguments
    partial_func = partial(
        process_user, 
        load_dir=load_dir, 
        output_dir=output_dir, 
        start_date=start_date, 
        end_date=end_date 
    )
    
    multiprocessing.set_start_method("fork")  # 'spawn' on Windows/macOS
    results = process_map(partial_func, all_users, max_workers=os.cpu_count(), chunksize=10)

    print("all_done")