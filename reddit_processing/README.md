## Raw Reddit Processing

The scripts in this directory process the raw dumps from Reddit into user-level trajectories and related data.  
They should be executed sequentially in the numeric order shown below.

- `1_extract_user_posts.py`  
  Extracts posts of the users of interest.

- `2_parent_reply_posts.py`  
  Extracts parent and reply posts to the posts of the users of interest.

- `2.5_collate_parent_reply_posts.py`  
  Collates the parent and reply posts into a single dataset.

- `3_extract_submissions.py`  
  Extracts submissions of the users of interest.

- `4_per_user_processing.py`  
  Processes posts and parent/replies, sorting them by individual user.

- `4.5_collate_per_user.py`  
  Collates per-user data into a single dataset.

- `5_agreement_classification.py`  
  Performs agreement classification for user replies, based on the parent post.

- `6_trajectory_construction.py`  
  Constructs trajectories for each user.

- `7_create_sample_matched_df_w_perturb.ipynb`  
  Creates a dataframe with noise perturbations (including the no-perturbation baseline).

- `8_content_embeddings/`  
  Contains scripts that embed user content in posts and submissions using ModernBERT.  
  These were run on a high-performance computing cluster due to the computational load.

- `9_create_first_n_df.ipynb`  
  Creates a dataframe based on the first *N* activities of users  
  (used in `experiments/classification/early_detection`).