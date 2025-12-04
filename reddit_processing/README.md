The scripts in this directory processes the raw dumps from Reddit. The scripts should be executed sequentially.

-----
|1_extract_user_posts.py                      | extracts posts of interested users
|2_parent_reply_posts.py                      | extracts parent and reply posts to the posts of the interested users
|2.5_collate_parent_reply_posts.py            | collation
|3_extract_submissions.py                     | extract submissions of interested users
|4_per_user_processing.py                     | processes the posts and parent/replies, sorting them by individual user
|4.5_collate_per_user.py                      | collation
|5_agreement_classification.py                | agreement classification for user replies based on parent post
|6_trajectory_construction.py                 | constructs trajectories for each user
|7_create_sample_matched_df_w_perturb.ipynb   | creates dataframe with noise perturbations including no perturbations
|8_content_embeddings                         | this folder contains scripts that embeds user content in posts and submissions using modernbert. These were run on a high performance computing cluster due to the high computation load.
|9_create_first_n_df.ipynb                    | creates dataframe based on the first N activities of users. for use in experiments in classification/early_detection
|10_simulate_trajectories_for_evasion.ipynb   | generates simulated account hijacking trajectories. For use in experiments in classification/detection_evade