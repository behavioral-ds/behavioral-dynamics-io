import json
import os
import pickle
import pandas as pd
from archive_processing_utils import load_df_jsonl
from multiprocessing import Pool, set_start_method


def collate_user(user, processed_dir_path, output_dir):
    
    user_dir_path = os.path.join(processed_dir_path, user)
    if os.path.isdir(user_dir_path):
        user_output_dir_path = os.path.join(output_dir, user)
        os.makedirs(user_output_dir_path, exist_ok=True)

        # get all files in user dir
        user_files = [f for f in os.listdir(user_dir_path) if os.path.isfile(os.path.join(user_dir_path, f))]
        
        user_submission_df_list = []
        user_comment_submission_df_list = []
        user_comment_reply_df_list = []
        parent_comments_df_list = []
        reply_comments_df_list = []

        for file in user_files:
            file_path = os.path.join(user_dir_path, file)

            if file.startswith("submission_"):
                df = pd.read_csv(file_path, dtype=str, lineterminator='\n')
                user_submission_df_list.append(df)
            elif file.startswith("user_comment_submission_"):
                df = pd.read_csv(file_path, dtype=str, lineterminator='\n')
                user_comment_submission_df_list.append(df)
            elif file.startswith("user_comment_reply_"):
                df = pd.read_csv(file_path, dtype=str, lineterminator='\n')
                user_comment_reply_df_list.append(df)
            elif file.startswith("parent_comments_"):
                df = pd.read_csv(file_path, dtype=str, lineterminator='\n')
                parent_comments_df_list.append(df)
            elif file.startswith("reply_comments_"):
                df = pd.read_csv(file_path, dtype=str, lineterminator='\n')
                reply_comments_df_list.append(df)
            else:
                print(file, "not processed")


        if len(user_submission_df_list) > 0:
            user_submission_df = pd.concat(user_submission_df_list, ignore_index=True)
            path = os.path.join(user_output_dir_path, f"all_submission.csv")
            user_submission_df.to_csv(path)

        if len(user_comment_submission_df_list) > 0:
            user_comment_submission_df = pd.concat(user_comment_submission_df_list, ignore_index=True)
            path = os.path.join(user_output_dir_path, f"all_user_comment_submission.csv")
            user_comment_submission_df.to_csv(path)

        if len(user_comment_reply_df_list) > 0:
            user_comment_reply_df = pd.concat(user_comment_reply_df_list, ignore_index=True)
            path = os.path.join(user_output_dir_path, f"all_user_comment_reply.csv")
            user_comment_reply_df.to_csv(path)

        if len(parent_comments_df_list) > 0:
            parent_comments_df = pd.concat(parent_comments_df_list, ignore_index=True)
            path = os.path.join(user_output_dir_path, f"all_parent_comments.csv")
            parent_comments_df.to_csv(path)

        if len(reply_comments_df_list) > 0:
            reply_comments_df = pd.concat(reply_comments_df_list, ignore_index=True)
            path = os.path.join(user_output_dir_path, f"all_reply_comments.csv")
            reply_comments_df.to_csv(path)
    else:
        print(user, "doesnt exist (probably did not fit in the timeframe)")


if __name__ == "__main__":


    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = int(config["collection_start_date"])
        collection_end_date = int(config["collection_end_date"])
        root_output_dir_path = config["root_output_dir_path"]
        interested_usernames_path = config["interested_usernames_path"]
        processed_per_user_path =  root_output_dir_path + "/4_per_user_processing" 
    
    with open(interested_usernames_path, "rb") as f:
        interested_users = set(pickle.load(f))
    print(interested_users)
    print(len(interested_users))

    output_dir = root_output_dir_path + "/4_per_user_processing_collated"
    os.makedirs(output_dir, exist_ok=True)

    def collate_user_wrapper(user):
        return collate_user(user, processed_per_user_path, output_dir)

    with Pool() as pool:
        pool.map(collate_user_wrapper, interested_users)
        
    print("all done")

