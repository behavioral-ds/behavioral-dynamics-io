import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
from collections import defaultdict

import reddit_processing_utils
import irl_utils

def get_state_action_names_12_6():
    states = []
    actions = [
        "WR", # wait reply
        "CT", # create thread
        "RC", # root comment
        "PR+", # post postitive reply
        "PR~", # post neutral reply
        "PR-" # post negative reply
    ]


    state_components = [
        "GR+", # get positive reply
        "GR~", # get neutral reply
        "GR-", # get negative reply
        "I", # initial flag
        "E", # engaged flag
        "T", # create thread
        "RC", # root comment
        "R+", # post positive reply
        "R~", # post neutral reply
        "R-", # post negative reply
    ]

    # dict of all possible legal states composed of state components
    state_dict = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        4: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        5: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        6: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        7: [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        8: [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        9: [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        10: [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        11: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }


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
    # 
    for v in state_dict.values():
        active_indices = np.where(np.array(v) == 1)[0]
        state_name = ""
        for i in active_indices:
            state_name = state_name + state_components[i]
        states.append(state_name)
    return states, actions

def activity_to_traj(df_total):
    """_summary_

    Args:
        df_total (_type_): _description_

    Returns:
        _type_: _description_
    """

    a_dict = {}

    all_action_it = [
        0,  # Create new thread
        0,  # Post root comment (neut)
        0,  # Post reply comment (agree)
        0,  # Post reply comment (neut)
        0,  # Post reply comment (disagree)
    ]

    dict_key_counter = 1
    a_dict[0] = all_action_it
    for a in range(len(all_action_it)):
        action = all_action_it.copy()
        action[a] = 1
        a_dict[dict_key_counter] = action
        dict_key_counter += 1
    s_dict = {}

    state_list = []

    reply_it = [
        0,  # Get reply (agree)
        0,  # Get reply (neut)
        0,  # Get reply (disagree)
    ]

    interact_it = [
        0,  # First interaction
        0,  # Already interacted
    ]


    create_thread_it = [
        0,  # Create new thread
    ]
    action_it = [
        0,  # Post root comment (neut)
        0,  # Post reply comment (agree)
        0,  # Post reply comment (neut)
        0,  # Post reply comment (disagree)
    ]

    for r in range(len(reply_it)):
        reply = [
            0,  # Get reply (agree)
            0,  # Get reply (neut)
            0,  # Get reply (disagree)
        ]
        reply[r] = 1
        state_list.append(reply + interact_it  + create_thread_it + action_it)

    # create thread states
    interact = [
        0,  # First interaction
        0,  # Already interacted
    ]
    interact[0] = 1
    state_list.append(reply_it + interact + [1] + action_it)

    for i in range(len(interact_it)):
        for a in range(len(action_it)):
            interact = [
                0,  # First interaction
                0,  # Already interacted
            ]

            action = [
                0,  # Post root comment (neut)
                0,  # Post reply comment (agree)
                0,  # Post reply comment (neut)
                0,  # Post reply comment (disagree)
            ]

            interact[i] = 1
            action[a] = 1
            state_list.append(reply_it + interact + create_thread_it  + action )

    dict_key_counter = 0
    for state in state_list:
        s_dict[dict_key_counter] = state
        dict_key_counter += 1



    state_counter = np.zeros(len(s_dict))

    feature_matrix = []  

    for el in s_dict:
        array = s_dict[el]
        int_array = [int(s) for s in array]
        feature_matrix.append(int_array)
    feature_matrix = np.asarray(feature_matrix)
    c_dict = {}
    for i in np.arange(len(s_dict)):
        c_dict[i] = 0

    n_actions = len(a_dict)
    n_states = len(s_dict)

    threads = df_total["thread_id"].unique()
    trajectories = []
    state_sequence = []

    # drop duplicate entries
    cols_to_match = ["created_utc", "subreddit", "id","body", "title","subreddit_id","parent_id"]
    cols_to_match = [col for col in cols_to_match if col in df_total.columns]
    df_total = df_total.drop_duplicates(subset=cols_to_match, keep='first')
    
    # trajectory length
    default_length = 50  # set as a default value, it can be edited based on users' activity
    if len(df_total) > default_length:
        trajectory_length = default_length
    else:
        trajectory_length = len(df_total) - 1

    thread_lengths = defaultdict(int)

    df_total["first_active"] = False
    df_total["last_active"] = False
    
    
    
    # go through each thread and set first and last interacted
    for thread_id in threads:
        df_thread = df_total[df_total["thread_id"] == thread_id].copy()
        thread_lengths[len(df_thread)] += 1

        
        df_thread.sort_values(by=['created_utc', "action"], inplace=True)

        # Set the "end" value of the last active row to true
        try:
            first_active_row_index = df_thread[df_thread["action_type"] == "active"].index[0]
            df_total.at[first_active_row_index, "first_active"] = True

            last_active_row_index = df_thread[df_thread["action_type"] == "active"].index[-1]
            df_total.at[last_active_row_index, "last_active"] = True
        except Exception as e:
            None


    period_traj = []
    initial_state = df_total.iloc[0]

    # state will always be first interaction as it is the first action in the user's history

    first_active = 0
    if initial_state["first_active"]:
        first_active = 1

    state_ = [
        0,  # Get reply (agree)
        0,  # Get reply (neut)
        0,  # Get reply (disagree)
        first_active,  # First interaction
        0,  # Already interacted
       
    ]

    if (initial_state.action == 1):
        action_features_0 = np.asarray([
                1,  # Create new thread
                0,  # Post root comment (neut)
                0,  # Post reply comment (agree)
                0,  # Post reply comment (neut)
                0,  # Post reply comment (disagree)
            ])
    elif (initial_state.action == 2):
        split = initial_state.parent_id.split("_")
        if split[0] == "t3":
            action_features_0 = np.asarray([
                0,  # Create new thread
                1,  # Post root comment
                0,  # Post reply comment (agree)
                0,  # Post reply comment (neut)
                0,  # Post reply comment (disagree)
            ])
        else:
            # check
            if initial_state.agree_prediciton == "0": # disagree
                action_features_0 = np.asarray([
                    0,  # Create new thread
                    0,  # Post root comment (neut)
                    0,  # Post reply comment (agree)
                    0,  # Post reply comment (neut)
                    1,  # Post reply comment (disagree)
                ])
            elif initial_state.agree_prediciton == "2": # agree
                action_features_0 = np.asarray([
                    0,  # Create new thread
                    0,  # Post root comment (neut)
                    1,  # Post reply comment (agree)
                    0,  # Post reply comment (neut)
                    0,  # Post reply comment (disagree)
                ])
            else: # neutral
                action_features_0 = np.asarray([
                    0,  # Create new thread
                    0,  # Post root comment (neut)
                    0,  # Post reply comment (agree)
                    1,  # Post reply comment (neut)
                    0,  # Post reply comment (disagree)
                ])
    elif (initial_state.action == 3):
        if initial_state.parent_id is np.nan:
            action_features_0 = np.asarray([
                0,  # Create new thread
                1,  # Post root comment
                0,  # Post reply comment (agree)
                0,  # Post reply comment (neut)
                0,  # Post reply comment (disagree)
            ])
        else:
            split = initial_state.parent_id.split("_")
            if split[0] == "t3":
                action_features_0 = np.asarray([
                    0,  # Create new thread
                    1,  # Post root comment
                    0,  # Post reply comment (agree)
                    0,  # Post reply comment (neut)
                    0,  # Post reply comment (disagree)
                ])
            else:
                # check
                if initial_state.agree_prediciton == "0":  # disagree
                    action_features_0 = np.asarray([
                        0,  # Create new thread
                        0,  # Post root comment (neut)
                        0,  # Post reply comment (agree)
                        0,  # Post reply comment (neut)
                        1,  # Post reply comment (disagree)
                    ])
                elif initial_state.agree_prediciton == "2":  # agree
                    action_features_0 = np.asarray([
                        0,  # Create new thread
                        0,  # Post root comment (neut)
                        1,  # Post reply comment (agree)
                        0,  # Post reply comment (neut)
                        0,  # Post reply comment (disagree)
                    ])
                else:  # neutral
                    action_features_0 = np.asarray([
                        0,  # Create new thread
                        0,  # Post root comment (neut)
                        0,  # Post reply comment (agree)
                        1,  # Post reply comment (neut)
                        0,  # Post reply comment (disagree)
                    ])
    else:
        # shouldnt really reach this state but it can happen if weird stuff occured in the scraping
        action_features_0 = None
        
        state_ = None
        print("started in bad state somehow")



    def add_state_action(state, action):
        state_counter[state] += 1
        c_dict[state] += 1
        period_traj.append([state, action])
        state_sequence.append([state, action])

    def map_state_action_to_feature(action_feature, state_feature):
        if action_feature is not None and state_feature is not None:
            action_t = np.where(action_feature > 0, 1, 0)
            action = list(a_dict.keys())[list(a_dict.values()).index(list(action_t))]  # python3

            feature_vec = np.concatenate((state_feature, action_t), axis=0)
            feature = list(s_dict.keys())[list(s_dict.values()).index(list(feature_vec))]


            return feature, action # returns the state that taking the action entered into
        return None, None
    
    
    prev_state = None
    try:
        prev_state, _ = map_state_action_to_feature(action_features_0, state_)
    except Exception as e:
        print(e)


    #
    # # if there is an action, check upvote karma status
    for t in np.arange(len(df_total))[1:]:
        current_timestep = df_total.iloc[t]
        if (current_timestep.action == 4):
            action_features_0 = np.asarray([0,  # Create new thread
                                            0,  # Post root comment
                                            0,  # Post reply comment (agree)
                                            0,  # Post reply comment (neut)
                                            0,  # Post reply comment (disagree)
                                            ])

            # check
            if current_timestep.agree_prediciton == "0":  # disagree
                state_ = [
                    0,  # Get reply (agree)
                    0,  # Get reply (neut)
                    1,  # Get reply (disagree)
                    0,  # First interaction
                    0,  # Already interacted
                ]
            elif current_timestep.agree_prediciton == "2":  # agree
                state_ = [
                    1,  # Get reply (agree)
                    0,  # Get reply (neut)
                    0,  # Get reply (disagree)
                    0,  # First interaction
                    0,  # Already interacted
                ]
            else:  # neutral
                state_ = [
                        0,  # Get reply (agree)
                        1,  # Get reply (neut)
                        0,  # Get reply (disagree)
                        0,  # First interaction
                        0,  # Already interacted
                ]
        else:
            if (current_timestep.action == 1):
                # this state should never be reached to be honest
                action_features_0 = np.asarray([1,  # Create new thread
                                                0,  # Post root comment
                                                0,  # Post reply comment (agree)
                                                0,  # Post reply comment (neut)
                                                0,  # Post reply comment (disagree)
                                                ])
            elif (current_timestep.action == 2):
                split = current_timestep.parent_id.split("_")
                if split[0] == "t3":
                    action_features_0 = np.asarray([0,  # Create new thread
                                                    1,  # Post root comment
                                                    0,  # Post reply comment (agree)
                                                    0,  # Post reply comment (neut)
                                                    0,  # Post reply comment (disagree)
                                                    ])
                else:

                    # check
                    if current_timestep.agree_prediciton == "0":  # disagree
                        action_features_0 = np.asarray([
                            0,  # Create new thread
                            0,  # Post root comment
                            0,  # Post reply comment (agree)
                            0,  # Post reply comment (neut)
                            1,  # Post reply comment (disagree)
                        ])
                    elif current_timestep.agree_prediciton == "2":  # agree
                        action_features_0 = np.asarray([
                            0,  # Create new thread
                            0,  # Post root comment
                            1,  # Post reply comment (agree)
                            0,  # Post reply comment (neut)
                            0,  # Post reply comment (disagree)
                        ])
                    else:  # neutral
                        action_features_0 = np.asarray([
                            0,  # Create new thread
                            0,  # Post root comment
                            0,  # Post reply comment (agree)
                            1,  # Post reply comment (neut)
                            0,  # Post reply comment (disagree)
                        ])
            elif (current_timestep.action == 3):
                if current_timestep.parent_id is np.nan:
                    action_features_0 = np.asarray([0,  # Create new thread
                                                    1,  # Post root comment
                                                    0,  # Post reply comment (agree)
                                                    0,  # Post reply comment (neut)
                                                    0,  # Post reply comment (disagree)
                                                    ])

                else:
                    split = current_timestep.parent_id.split("_")
                    if split[0] == "t3":
                        action_features_0 = np.asarray([0,  # Create new thread
                                                        1,  # Post root comment
                                                        0,  # Post reply comment (agree)
                                                        0,  # Post reply comment (neut)
                                                        0,  # Post reply comment (disagree)
                                                        ])
                    else:
                        # check
                        if current_timestep.agree_prediciton == "0":  # disagree
                            action_features_0 = np.asarray([
                                0,  # Create new thread
                                0,  # Post root comment
                                0,  # Post reply comment (agree)
                                0,  # Post reply comment (neut)
                                1,  # Post reply comment (disagree)
                            ])
                        elif current_timestep.agree_prediciton == "2":  # agree
                            action_features_0 = np.asarray([
                                0,  # Create new thread
                                0,  # Post root comment
                                1,  # Post reply comment (agree)
                                0,  # Post reply comment (neut)
                                0,  # Post reply comment (disagree)
                            ])
                        else:  # neutral
                            action_features_0 = np.asarray([
                                0,  # Create new thread
                                0,  # Post root comment
                                0,  # Post reply comment (agree)
                                1,  # Post reply comment (neut)
                                0,  # Post reply comment (disagree)
                            ])
            else:
                print("something is amiss",current_timestep)

            # determine state
            first_active = 0
            last_active = 0
            alread_interacted = 0

            if current_timestep["first_active"] == True:
                first_active = 1
            if current_timestep["last_active"] == True:
                last_active = 1
            if first_active == 0: #and last_active == 0:
                alread_interacted = 1

                
            state_ = [
                0,  # Get reply (agree)
                0,  # Get reply (neut)
                0,  # Get reply (disagree)
                first_active,  # First interaction
                alread_interacted,  # Already interacted
            ]

        try:
            new_state, action = map_state_action_to_feature(action_features_0, state_)
            if prev_state is not None:
                add_state_action(prev_state, action)
            prev_state = new_state
            
        except Exception as e:
            print(e)


        if len(period_traj) > 0 and len(period_traj) >= trajectory_length :

            trajectories.append(period_traj[:trajectory_length])  # trajectory story
            if len(period_traj) > trajectory_length:
                period_traj = period_traj[trajectory_length:]
                print("period_traj", period_traj)
            else:
                period_traj = []


    trajectories = np.asarray(trajectories)
    return trajectories, state_sequence, n_states, n_actions, feature_matrix, c_dict



def construct_tp_traj(user,
            all_user_dir,
            output_dir=None,
            start_date = None,
            end_date = None,
            threshold_active  = 5,
            threshold_passive = 5
        ):
    """ constructs the trajectories and tp for a given user

    Args:
        user (str): username of user 
        all_user_dir (_type_): path to dir containing all users, each in their own folder with the folder name matching the username
        output_dir (_type_, optional): _description_. Defaults to None.
        start_date (_type_, optional): _description_. Defaults to None.
        end_date (_type_, optional): _description_. Defaults to None.
        threshold_active (int, optional): _description_. Defaults to 5.
        threshold_passive (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    df_slice = get_user_activity_in_timeframe(user,
                all_user_dir,
                start_date=start_date,
                end_date=end_date)
    
    if df_slice is not None and len(df_slice) >= threshold_active + threshold_passive:
        # IRL
        # compute trajectories and other information
        trajectories, state_sequence, n_states, n_actions, feature_matrix, c_dict  = activity_to_traj(df_slice)
        
        # compute transition probabilities
        tp = irl_utils.compute_tp(state_sequence, n_states, n_actions)
        traj_counts = irl_utils.compute_state_count(state_sequence, n_states, n_actions)
        
        if output_dir is not None:
            np.savez(output_dir + "/" + user + ".npz", tp=tp, traj_count=traj_counts, traj=trajectories)

        return tp, traj_counts, trajectories, feature_matrix
    return None,None,None, None

def get_user_activity_in_timeframe(user,
            all_user_dir,
            start_date = None,
            end_date = None):
    """_summary_

    Args:
        user (_type_): _description_
        all_user_dir (_type_): _description_
        output_dir (_type_, optional): _description_. Defaults to None.
        start_date (_type_, optional): _description_. Defaults to None.
        end_date (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    user_path = all_user_dir + "/" + user
    
    user_submissions = pd.DataFrame()
    user_comment_reply = pd.DataFrame()
    user_comment_submission = pd.DataFrame()
    parent_comments = pd.DataFrame()
    reply_comments = pd.DataFrame()
    
    if os.path.exists(user_path + "/all_submission.csv"):
        user_submissions = pd.read_csv(user_path + "/all_submission.csv", dtype=str, lineterminator='\n')
        # print("len(user_submissions", len(user_submissions))
    if os.path.exists(user_path + "/all_user_comment_reply_w_agreement.csv"):
        user_comment_reply = pd.read_csv(user_path + "/all_user_comment_reply_w_agreement.csv", dtype=str, lineterminator='\n')
        # print("len(user_comment_reply", len(user_comment_reply))
    if os.path.exists(user_path + "/all_user_comment_submission.csv"):
        user_comment_submission = pd.read_csv(user_path + "/all_user_comment_submission.csv", dtype=str, lineterminator='\n')
        # print("len(user_comment_submission", len(user_comment_submission))
        
        
    if os.path.exists(user_path + "/all_parent_comments.csv"):
        parent_comments = pd.read_csv(user_path + "/all_parent_comments.csv", dtype = str, lineterminator='\n')
        # print("len(parent_comments", len(parent_comments))
        
    if os.path.exists(user_path + "/all_reply_comments_w_agreement.csv"):
        reply_comments = pd.read_csv(user_path + "/all_reply_comments_w_agreement.csv", dtype = str, lineterminator='\n')
        # print("len(reply_comments", len(reply_comments))
    
    user_submissions["action"] = 1
    user_comment_reply["action"] = 2
    user_comment_submission["action"] = 3
    
    parent_comments["action"] = 4
    reply_comments["action"] = 4
    
    
    def split_link_id(link_id):
        if isinstance(link_id, str):  # Check if the value is a string
            parts = link_id.split("_")
            if len(parts) > 1:
                return parts[1]
        return np.nan
    
    if len(user_submissions) > 0:
        user_submissions["thread_id"] = user_submissions["id"]
    if len(user_comment_reply) > 0:
        user_comment_reply["thread_id"] = user_comment_reply["link_id"].apply(split_link_id)
    if len(user_comment_submission) > 0:
        user_comment_submission["thread_id"] = user_comment_submission["link_id"].apply(split_link_id)
    if len(parent_comments) > 0:
        parent_comments["thread_id"] = parent_comments["link_id"].apply(split_link_id)
    if len(reply_comments) > 0:
        reply_comments["thread_id"] = reply_comments["link_id"].apply(split_link_id)
    
    num_active_actions = len(user_comment_reply) + len(user_comment_submission) + len(user_submissions)
    num_passive_actions = len(reply_comments) #+ len(parent_comments)
    
        
    # creating actions dataframe
    df_active_u = pd.concat([user_comment_reply, user_comment_submission, user_submissions, reply_comments], ignore_index=True)


    if len(df_active_u) <=0 or "created_utc" not in df_active_u.columns or "action" not in df_active_u.columns:
        return None

    df_active_u = df_active_u.sort_values(by=['created_utc',"action"]).copy()

    df_active_u.reset_index(inplace = True)
    del df_active_u['index']
    df_active_u["action_type"] = "active"
    # creating states dataframe
    df_passive_u = reply_comments #.append(user_comment_submission,ignore_index=True)
    if len(df_passive_u) > 0:
        df_passive_u = df_passive_u.sort_values(by=['created_utc',"action"])
        df_passive_u.reset_index(inplace = True)
        del df_passive_u['index']
        df_passive_u["action_type"] = "passive"
    
    
    # create total dataframe
    df_total = reddit_processing_utils.merge_df(df_active_u, df_passive_u)
    df_total['created_utc'] = pd.to_datetime(pd.to_numeric(df_total['created_utc']), unit='s')

    # drop duplicate entries
    cols_to_match = ["created_utc", "subreddit", "id", "body", "title", "subreddit_id", "parent_id"]
    # Filter out columns that don't exist in the DataFrame
    cols_to_match = [col for col in cols_to_match if col in df_total.columns]

    df_total = df_total.drop_duplicates(subset=cols_to_match, keep='first')

    if start_date is not None and end_date is not None:
        start_date_datetime = pd.to_datetime(start_date)
        end_date_datetime = pd.to_datetime(end_date)
        df_slice = df_total[(df_total['created_utc'] >= start_date_datetime) & (df_total['created_utc'] < end_date_datetime)].copy()
    else:
        df_slice = df_total.copy()
    # Reset the index 
    df_slice.reset_index(drop=True, inplace=True)

    return df_slice