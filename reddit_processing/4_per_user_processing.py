import json
import os
import pickle
import pandas as pd
from archive_processing_utils import load_df_jsonl
from multiprocessing import Pool, set_start_method

# Global variables to be shared via fork
parent_df = None
reply_df = None
interested_users = None
processed_comment_path_global = None
processed_submission_path_global = None
output_dir = None


def init_globals(processed_parent_reply_path, processed_comment_path, processed_submission_path, interested_users_path,output_dir):
    global parent_df, reply_df, interested_users, processed_comment_path_global, processed_submission_path_global

    parent_pickle_path = os.path.join(processed_parent_reply_path, "parent.pkl")
    reply_pickle_path = os.path.join(processed_parent_reply_path, "reply.pkl")

    processed_comment_path_global = processed_comment_path
    processed_submission_path_global = processed_submission_path

    parent_df = pd.read_pickle(parent_pickle_path)
    print("loaded parent df")

    reply_df = pd.read_pickle(reply_pickle_path)
    reply_df["parent_id_clean"] = reply_df["parent_id"].str.split("_").str[1]
    print("loaded reply df")

    with open(interested_users_path, "rb") as f:
        interested_users = set(pickle.load(f))


def process_month(args):
    year, month = args
    global parent_df, reply_df, interested_users

    print(f"Processing {year}-{month}")

    # Load data for the month

    comment_df = load_df_jsonl(f"{processed_comment_path_global}/RC_{year}-{month}")
    submission_df = load_df_jsonl(f"{processed_submission_path_global}/RS_{year}-{month}")

    comment_grouped = comment_df.groupby("author")
    submission_grouped = submission_df.groupby("author")

    for i, user in enumerate(interested_users):
        if i % 100 == 0:
            print(f"{year}-{month}: {i}/{len(interested_users)}")

        ### CHANGED: output_dir now comes from config
        user_folder_path = os.path.join(output_dir, user)

        ym_str = f"{year}_{month}"

        output_files = [
            f"submission_{ym_str}.csv",
            f"user_comment_submission_{ym_str}.csv",
            f"user_comment_reply_{ym_str}.csv",
            f"parent_comments_{ym_str}.csv",
            f"reply_comments_{ym_str}.csv"
        ]

        if all(os.path.exists(os.path.join(user_folder_path, f)) for f in output_files):
            continue

        user_comment_df = comment_grouped.get_group(user) if user in comment_grouped.groups else pd.DataFrame()
        user_submission_df = submission_grouped.get_group(user) if user in submission_grouped.groups else pd.DataFrame()

        user_comment_submission_df = pd.DataFrame()
        user_comment_reply_df = pd.DataFrame()
        if not user_comment_df.empty:
            user_comment_submission_df = user_comment_df[user_comment_df["parent_id"].str.startswith("t3_")]
            user_comment_reply_df = user_comment_df[user_comment_df["parent_id"].str.startswith("t1_")]

        if user_comment_df.empty and user_submission_df.empty:
            continue

        parent_comments_df = pd.DataFrame()
        if not user_comment_reply_df.empty:
            parent_ids = user_comment_reply_df["parent_id"].str.split("_").str[1]
            parent_comments_df = parent_df[parent_df["id"].isin(parent_ids)]

        reply_comments_df = pd.DataFrame()
        if not user_comment_df.empty:
            comment_ids = user_comment_df["id"]
            reply_comments_df = reply_df[reply_df["parent_id_clean"].isin(comment_ids)]

        if any(not df.empty for df in [user_submission_df, user_comment_submission_df, user_comment_reply_df, parent_comments_df, reply_comments_df]):
            os.makedirs(user_folder_path, exist_ok=True)

        if not user_submission_df.empty:
            path = os.path.join(user_folder_path, f"submission_{ym_str}.csv")
            if not os.path.exists(path):
                user_submission_df.to_csv(path)

        if not user_comment_submission_df.empty:
            path = os.path.join(user_folder_path, f"user_comment_submission_{ym_str}.csv")
            if not os.path.exists(path):
                user_comment_submission_df.to_csv(path)

        if not user_comment_reply_df.empty:
            path = os.path.join(user_folder_path, f"user_comment_reply_{ym_str}.csv")
            if not os.path.exists(path):
                user_comment_reply_df.to_csv(path)

        if not parent_comments_df.empty:
            path = os.path.join(user_folder_path, f"parent_comments_{ym_str}.csv")
            if not os.path.exists(path):
                parent_comments_df.to_csv(path)

        if not reply_comments_df.empty:
            path = os.path.join(user_folder_path, f"reply_comments_{ym_str}.csv")
            if not os.path.exists(path):
                reply_comments_df.to_csv(path)

    print(f"Done {year}-{month}")


if __name__ == "__main__":
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = int(config["collection_start_date"])
        collection_end_date = int(config["collection_end_date"])
        root_output_dir_path = config["root_output_dir_path"]
        interested_usernames_path = config["interested_usernames_path"]

        processed_comment_path = root_output_dir_path + "1_extract_user_posts"
        processed_parent_reply_path = root_output_dir_path + "2_extract_parent_reply"
        processed_submission_path = root_output_dir_path + "3_extract_submissions"

        output_dir = root_output_dir_path + "/4_per_user_processing"

    os.makedirs(output_dir, exist_ok=True)

    init_globals(
        processed_parent_reply_path,
        processed_comment_path=processed_comment_path,
        processed_submission_path=processed_submission_path,
        interested_users_path=interested_usernames_path,
        output_dir=output_dir
    )

    year_months = [
        (str(y), str(m).zfill(2)) 
        for y in range(collection_end_date - 1, collection_start_date - 1, -1) 
        for m in range(12, 0, -1)
    ]

    print(year_months)

    with Pool(processes=12) as pool:
        pool.map(process_month, year_months)

    print("All done.")
