import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from joblib import Parallel, delayed

def tokenize_input_sentences(sent_1, sent_2, tokenizer, max_seq_len = 512):
    try:
        tokens_1 = tokenizer.tokenize(sent_1)
        tokens_2 = tokenizer.tokenize(sent_2)
    except:
        print("sent_1",sent_1)
        print("sent_2",sent_2)
    # Truncate long sequence
    tokens_1 = tokens_1[:int(max_seq_len/2) - 2]
    tokens_2 = tokens_2[:int(max_seq_len/2) - 1]
    # Add special tokens to the `tokens`
    tokens = ['[CLS]'] + tokens_1 + ['[SEP]'] + tokens_2 + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*len(input_ids)
    # padding
    paddings = max_seq_len - len(input_ids)
    input_ids = input_ids + [0]*paddings
    input_mask = input_mask + [0]*paddings
    
    features = {
    'input_ids': input_ids,
    'attention_mask': input_mask 
    }
    return features

# Global cache dictionary for tokenization results
_tokenize_cache = {}

def cached_tokenize(sentence, tokenizer):
    if sentence not in _tokenize_cache:
        _tokenize_cache[sentence] = tokenizer.tokenize(sentence)
    return _tokenize_cache[sentence]


def tokenize_pair(sent_1, sent_2, tokenizer, max_seq_len=512):
    tokens_1 = cached_tokenize(sent_1, tokenizer)[:int(max_seq_len / 2) - 2]
    tokens_2 = cached_tokenize(sent_2, tokenizer)[:int(max_seq_len / 2) - 1]

    tokens = ['[CLS]'] + tokens_1 + ['[SEP]'] + tokens_2 + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    padding_len = max_seq_len - len(input_ids)
    input_ids += [0] * padding_len
    attention_mask += [0] * padding_len

    return input_ids, attention_mask


def tokenize_custom_batch(sent_1_list, sent_2_list, tokenizer, max_seq_len=512):
    half_len_1 = max_seq_len // 2 - 2
    half_len_2 = max_seq_len // 2 - 1

    # Batch tokenize both sentence lists
    tokens_1_batch = tokenizer.batch_encode_plus(
        sent_1_list,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )["input_ids"]

    tokens_2_batch = tokenizer.batch_encode_plus(
        sent_2_list,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )["input_ids"]

    input_ids_list = []
    attention_mask_list = []

    for tokens_1, tokens_2 in zip(tokens_1_batch, tokens_2_batch):
        tokens_1 = tokens_1[:half_len_1]
        tokens_2 = tokens_2[:half_len_2]

        input_ids = [tokenizer.cls_token_id] + tokens_1 + [tokenizer.sep_token_id] + tokens_2 + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        padding_len = max_seq_len - len(input_ids)
        if padding_len > 0:
            input_ids += [tokenizer.pad_token_id] * padding_len
            attention_mask += [0] * padding_len

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
    }

def parent_reply_lookup(parent_df,reply_df):
    # Reset to simple 0…N‑1 index
    reply_df = reply_df.reset_index(drop=True)

    # Keep a copy of that index so we can retrieve it after the merge
    reply_df["orig_idx"] = reply_df.index

    # Clean parent_id (strip the prefix before the underscore)
    reply_df["parent_id_clean"] = reply_df["parent_id"].str.split("_").str[1]

    # Merge, but don’t let pandas reorder rows (sort=False keeps left‑side order)
    merged_df = reply_df.merge(
        parent_df[["id", "body"]],
        how="left",
        left_on="parent_id_clean",
        right_on="id",
        suffixes=("_reply", "_parent"),
        sort=False                     # ← key for preserving order
    )

    # Mask: both parent & reply text must be strings
    valid_mask = (
        merged_df["body_parent"].apply(lambda x: isinstance(x, str)) &
        merged_df["body_reply"].apply(lambda x: isinstance(x, str))
    )

    parent_text_list  = merged_df.loc[valid_mask, "body_parent"].tolist()
    child_text_list   = merged_df.loc[valid_mask, "body_reply"].tolist()
    used_indices      = merged_df.loc[valid_mask, "orig_idx"].tolist()  
    no_parent_indices = merged_df.loc[~valid_mask, "orig_idx"].tolist() 

    return parent_text_list, child_text_list, used_indices, no_parent_indices

def process_parent_reply(model, parent_df,reply_df, tokenizer, device):
    parent_text_list, child_text_list, used_indices, no_parent_indices = parent_reply_lookup(parent_df,reply_df)

    if len(used_indices) == 0:
        return None

    classify_encoded_instances = tokenize_custom_batch(
        parent_text_list,
        child_text_list,
        tokenizer,
        max_seq_len=512
    )


    input_ids = torch.Tensor(classify_encoded_instances["input_ids"]) #.to(device)
    attention_masks = torch.Tensor(classify_encoded_instances["attention_mask"])#.to(device)

    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    counter = 0
    predictions = []
    # Classify the dataset in batches
    for batch in dataloader:
        counter +=1
        # print(str(counter*batch_size),"/",len(reply_df))
        batch_input_ids, batch_attention_masks = batch
        batch_input_ids = batch_input_ids.to(device).to(torch.int64)
        batch_attention_masks = batch_attention_masks.to(device)

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)

        predictions.extend(pred.tolist())


    # output
    reply_df_with_pred = pd.DataFrame(reply_df.iloc[used_indices])
    reply_df_with_pred["agree_prediciton"] = predictions
    return reply_df_with_pred

if __name__ == "__main__":

    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = int(config["collection_start_date"])
        collection_end_date = int(config["collection_end_date"])
        root_output_dir_path = config["root_output_dir_path"]
        deberta_weight_path = config["deberta_weight_path"]
        processed_per_user_path =  root_output_dir_path + "/4_per_user_processing" 

    model_name = "microsoft/deberta-v3-base"
    load_model_name = deberta_weight_path 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(load_model_name, num_labels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model= torch.nn.DataParallel(model)
    model = model.to(device)

    # go per user
    # predictions 0- disagree, 1- neutral, 2- agree
    user_dir = processed_per_user_path + "/4_per_user_processing_collated"
    
    batch_size = 400

    all_users = sorted(os.listdir(user_dir))
    done_counter = 0
    for user in all_users:
        print(user)
        parent_df = None
        user_reply_df = None
        user_submission_df = None
        reply_comments_df = None
        if os.path.exists(user_dir + "/" + user + "/all_parent_comments.csv") and os.path.exists(user_dir + "/" + user + "/all_user_comment_reply.csv") and not os.path.exists(user_dir + "/" + user + "/all_user_comment_reply_w_agreement.csv"):
            parent_df = pd.read_csv(user_dir + "/" + user + "/all_parent_comments.csv",dtype=str,lineterminator='\n')
            user_reply_df = pd.read_csv(user_dir + "/" + user + "/all_user_comment_reply.csv",dtype=str,lineterminator='\n')
                        
            user_reply_df_pred = process_parent_reply(model, parent_df,user_reply_df, tokenizer, device)
            if user_reply_df_pred is not None:
                user_reply_df_pred.to_csv(user_dir + "/" + user + "/all_user_comment_reply_w_agreement.csv")

        if os.path.exists(user_dir + "/" + user + "/all_user_comment_reply.csv") and os.path.exists(user_dir + "/" + user + "/all_reply_comments.csv") and not os.path.exists(user_dir + "/" + user + "/all_reply_comments_w_agreement.csv"):
            if user_reply_df is None:
                user_reply_df = pd.read_csv(user_dir + "/" + user + "/all_user_comment_reply.csv",dtype=str,lineterminator='\n')
            user_submission_df = pd.read_csv(user_dir + "/" + user + "/all_user_comment_submission.csv",dtype=str,lineterminator='\n')
            reply_comments_df = pd.read_csv(user_dir + "/" + user + "/all_reply_comments.csv",dtype=str,lineterminator='\n')
            
            
            reply_df_pred = process_parent_reply(model, pd.concat([user_reply_df, user_submission_df]),reply_comments_df, tokenizer, device)
            if reply_df_pred is not None:
                reply_df_pred.to_csv(user_dir + "/" + user + "/all_reply_comments_w_agreement.csv")
        done_counter += 1
        print(done_counter, "/", len(all_users))
    print("all done")
