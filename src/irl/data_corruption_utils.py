import numpy as np
import pandas as pd
import pickle
import os
import random
from irl_utils import compute_state_count, compute_tp
from collections import defaultdict 

def normalize_ignore_zeros(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.divide(matrix, row_sums, where=row_sums != 0)
    return normalized

def normalize_replace_zeros(matrix):
    matrix = np.asarray(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    n_cols = matrix.shape[1]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.divide(matrix, row_sums, where=row_sums != 0)
    # Identify rows where the sum is zero
    zero_sum_rows = (row_sums == 0).flatten()
    # Replace those rows with a uniform distribution
    normalized[zero_sum_rows] = 1.0 / n_cols
    return normalized


def matched_window(df1, df2, target_column="traj", column='traj_total', num_s=12, num_a=6):
    df2_sorted = df2.sort_values(by=column).reset_index(drop=True)
    matched_rows_df1 = []
    matched_rows_df2 = []
    used_indices_df2 = set()

    for _, row in df1.iterrows():
        target_val = int(row[column])

        # Get eligible rows in df2 which are at least as long as that of df1
        mask = (df2_sorted[column] >= target_val)
        eligible = df2_sorted[mask]
        eligible = eligible[~eligible.index.isin(used_indices_df2)]

        if not eligible.empty:
            selected = eligible.sample(1)
            selected_row = selected.iloc[0].copy()

            traj = selected_row.traj
            # reshape to select window
            reshaped_traj = traj.reshape(-1,2) 
            
            # Select random start index for window
            start_index = random.randint(0, int(selected_row[column]) - target_val)
            window = reshaped_traj[start_index:int(start_index + target_val)]

            traj_window = reshape_to_n_50_2(window)
            tp_window = compute_tp(traj_window.reshape(-1,2) , num_s, num_a)            
            traj_counts_window =  compute_state_count(traj_window.reshape(-1,2) , num_s, num_a)

            selected_row["traj_window"] = traj_window
            selected_row["tp_window"] = tp_window
            selected_row["traj_counts_window"] = traj_counts_window
            selected_row['traj_total_window'] = np.sum(traj_counts_window)


            matched_rows_df1.append(row)
            matched_rows_df2.append(selected_row)
            used_indices_df2.add(selected.index[0])

    df1_matched = pd.DataFrame(matched_rows_df1).reset_index(drop=True)
    df2_matched = pd.DataFrame(matched_rows_df2).reset_index(drop=True)
    return df1_matched, df2_matched


def matched_sampling_with_tolerance(df1, df2, column='traj_total', r=1):
    df2_sorted = df2.sort_values(by=column).reset_index(drop=True)
    matched_rows_df1 = []
    matched_rows_df2 = []
    used_indices_df2 = set()

    for _, row in df1.iterrows():
        target_val = row[column]
        
        # Get eligible rows in df2 within ±r and not already used
        mask = (df2_sorted[column] >= target_val - r) & (df2_sorted[column] <= target_val + r)
        eligible = df2_sorted[mask]
        eligible = eligible[~eligible.index.isin(used_indices_df2)]

        if not eligible.empty:
            selected = eligible.sample(1)
            matched_rows_df1.append(row)
            matched_rows_df2.append(selected.iloc[0])
            used_indices_df2.add(selected.index[0])
        # Else: skip unmatched

    df1_matched = pd.DataFrame(matched_rows_df1).reset_index(drop=True)
    df2_matched = pd.DataFrame(matched_rows_df2).reset_index(drop=True)

    return df1_matched, df2_matched

def matched_sampling_longer(df1, df2, column='traj_total'):
    
    df2_sorted = df2.sort_values(by=column).reset_index(drop=True)
    matched_rows_df1 = []
    matched_rows_df2 = []
    used_indices_df2 = set()

    for _, row in df1.iterrows():
        target_val = row[column]
        
        # Get eligible rows in df2 within ±r and not already used
        mask = (df2_sorted[column] >= target_val )
        eligible = df2_sorted[mask]
        eligible = eligible[~eligible.index.isin(used_indices_df2)]

        if not eligible.empty:
            selected = eligible.sample(1)
            matched_rows_df1.append(row)
            matched_rows_df2.append(selected.iloc[0])
            used_indices_df2.add(selected.index[0])
        # Else: skip unmatched

    df1_matched = pd.DataFrame(matched_rows_df1).reset_index(drop=True)
    df2_matched = pd.DataFrame(matched_rows_df2).reset_index(drop=True)

    return df1_matched, df2_matched

def reshape_to_n_50_2(arr):
    # Do nothing if there is < 50 activities
    if arr.shape[0] < 50:
        return arr[np.newaxis, :, :]  # Shape (1, k, 2)
    total_rows = arr.shape[0]
    # Drop rows that don't fit into 50-row blocks
    usable_rows = (total_rows // 50) * 50
    trimmed = arr[:usable_rows]
    reshaped = trimmed.reshape(-1, 50, 2)
    return reshaped

def drop_traj(traj, drop_percent= 0.1):
    reshaped_traj = traj.reshape(-1,2)

    n_rows = reshaped_traj.shape[0]
    n_drop = int(n_rows * drop_percent)
    keep_indices = np.random.choice(n_rows, n_rows - n_drop, replace=False)
    return reshape_to_n_50_2(reshaped_traj[keep_indices])

def perturb_traj(arr, perturb_percent, num_states, num_actions, legal_transitions):
    
    arr = arr.copy().reshape(-1, 2)  # Avoid modifying the original array
    k = arr.shape[0]
    num_to_change = int(k * perturb_percent)

    # Choose indices to modify
    indices = np.sort(np.random.choice(k, num_to_change, replace=False))
    
    for i in indices:


        # action
        new_a = np.random.randint(0, num_actions)
        s = arr[i, 0]
        legal_next_s = np.argwhere(legal_transitions[s, new_a] == 1)
        new_s = legal_next_s[np.random.choice(legal_next_s.shape[0])][0]

        # set action
        arr[i, 1] = new_a

        # only change next state if we dont go beyond the end of the trajectory
        if i+1 < k:
            # set next state based on action
            arr[i+1, 0] = new_s

        if i == 0:
            # if we are at the start, then also generate a random start state
            arr[i, 0] = np.random.randint(0, num_states)

        
    return reshape_to_n_50_2(arr)


# def perturb_traj(arr, perturb_percent, num_states, num_actions, legal_transitions):
    
#     arr = arr.copy().reshape(-1, 2)  # Avoid modifying the original array
#     k = arr.shape[0]
#     num_to_change = int(k * perturb_percent)

#     # Choose indices to modify
#     indices = np.sort(np.random.choice(k, num_to_change, replace=False))
    
#     for i in indices:
#         if i > 0:
#             prev_s = arr[i-1, 0]
#             prev_a = arr[i-1, 1]
#             legal_next_s = np.argwhere(legal_transitions[prev_s,prev_a] == 1)
#             new_s = legal_next_s[np.random.choice(legal_next_s.shape[0])][0]
#         else:
#             # state
#             new_s = np.random.randint(0, num_states)
      
#         arr[i, 0] = new_s
        
#         # action
#         new_a = np.random.randint(0, num_actions)
#         arr[i, 1] = new_a
#     return reshape_to_n_50_2(arr)


def perturb_traj_diff(arr, perturb_percent, num_states, num_actions):    
    arr = arr.copy().reshape(-1, 2)  # Avoid modifying the original array
    k = arr.shape[0]
    num_to_change = int(k * perturb_percent)

    # Choose indices to modify
    indices = np.random.choice(k, num_to_change, replace=False)

    for i in indices:
        # state
        old_val = arr[i, 0]
        
        # Generate a new value in [0, n) that's not equal to old_val
        new_val = np.random.randint(0, num_states - 1)
        if new_val >= old_val:
            new_val += 1  # Shift up to avoid equality
        arr[i, 0] = new_val
        
        # action
        old_val = arr[i, 1]
        
        # Generate a new value in [0, n) that's not equal to old_val
        new_val = np.random.randint(0, num_actions - 1)
        if new_val >= old_val:
            new_val += 1  # Shift up to avoid equality
        arr[i, 1] = new_val
    return reshape_to_n_50_2(arr)

def perturb_traj_random(arr, perturb_percent, num_states, num_actions):
    arr = arr.copy().reshape(-1, 2)  # Avoid modifying the original array
    k = arr.shape[0]
    num_to_change = int(k * perturb_percent)

    # Choose indices to modify
    indices = np.random.choice(k, num_to_change, replace=False)

    for i in indices:
        # state
        new_s = np.random.randint(0, num_states)
        arr[i, 0] = new_s
        # action
        new_a = np.random.randint(0, num_actions)
        arr[i, 1] = new_a
    return reshape_to_n_50_2(arr)

def perturb_traj_state_only(arr, perturb_percent, max_value):
    arr = arr.copy().reshape(-1, 2)  # Avoid modifying the original array
    k = arr.shape[0]
    num_to_change = int(k * perturb_percent)

    # Choose indices to modify
    indices = np.random.choice(k, num_to_change, replace=False)

    for i in indices:
        old_val = arr[i, 0]
        # Generate a new value in [0, n) that's not equal to old_val
        new_val = np.random.randint(0, max_value - 1)
        if new_val >= old_val:
            new_val += 1  # Shift up to avoid equality
        arr[i, 0] = new_val
    return reshape_to_n_50_2(arr)

    
def randomize_with_same_sum(matrix):
    """
    generates a random matrix of the same size with the same sum 
    assuming values in the matrix are integers

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    shape = matrix.shape
    total = matrix.sum()
    size = matrix.size
    cuts = np.sort(np.random.choice(np.arange(1, total + size), size - 1, replace=False))
    values = np.diff(np.concatenate(([0], cuts, [total + size])))
    values = values - 1
    np.random.shuffle(values)
    return values.reshape(shape)


def match_traj_len(traj_1, traj_2):
    if len(traj_1) > len(traj_2):
        return None
    
    traj_1_reshaped = traj_1.reshape(-1,2)
    traj_2_reshaped = traj_2.reshape(-1,2)
    

    start_idx = np.random.randint(0, len(traj_2_reshaped) - len(traj_1_reshaped) + 1)
    
    return reshape_to_n_50_2(traj_2_reshaped[start_idx:start_idx + len(traj_1_reshaped)])


if __name__ == "__main__":
    a = np.zeros((2,50,2))
    b = np.zeros((5,50,2))
    z = match_traj_len(a, b)
    print(z.shape)

    a = np.zeros((1,43,2))
    b = np.zeros((2,50,2))
    z = match_traj_len(a, b)
    print(z.shape)


