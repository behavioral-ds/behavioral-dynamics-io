"""
utils for loading and processing output dfs
"""
import numpy as np
import pandas as pd

def parse_string_to_floats(string):
    # Remove square brackets and split the string
    elements = string.strip("[]\n").split()
    # Convert each element to a float
    floats = [float(element) for element in elements]
    return floats
    
def parse_policy(string):
    data = string.replace('\n', '').replace('[', '').replace(']', '').split()
    array = np.array(data, dtype=np.float32).reshape(-1, 6)
    return array

def load_reward_df_no_test(reward_df_path):
    """loads a reward df csv file that is the output of IRL and does not contain a "test" column 
    Args:
        reward_df_path (_type_): IRL output df csv
    Returns:
        _type_: loaded df
    """
    reward_df = pd.read_csv(reward_df_path)
    reward_df['r'] = reward_df['r'].apply(parse_string_to_floats)
    reward_df["policy"] =  reward_df['policy'].apply(parse_policy)
    reward_df["traj_counts"] =  reward_df['traj_counts'].apply(parse_policy)
    
    labels = ["user","policy","reward","traj_counts"]
    df = pd.DataFrame(columns=labels)
    data_list = np.array(reward_df["r"].values.tolist())

    for i in range(len(data_list)):
        row = data_list[i]
        e = len(df)
        df.loc[e] = [reward_df["screen_name"].values[i],
                     reward_df["policy"].values[i], 
                     reward_df["r"].values[i],
                     reward_df["traj_counts"].values[i]]
    return df


