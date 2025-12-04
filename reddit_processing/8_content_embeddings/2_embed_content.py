from transformers import AutoTokenizer, AutoModelForMaskedLM, ModernBertModel
import torch
from torch.nn import DataParallel
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Note: this script was run on a separate HPC cluster


def init_worker(model_id = "answerdotai/ModernBERT-large"):
    """Initialize model and tokenizer in each worker process."""
    #global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBertModel.from_pretrained(model_id)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
    model.to(device)
    return tokenizer, model
    

def get_bert_embedding_batch(
    texts,
    user_df,
    tokenizer,
    model,
    batch_size=128,
    save_dir="/modernbert_embed/",
    start_batch=0,
    end_batch=-1  
):
    """
    Get [CLS] embeddings for a list of texts using batching, and save each batch immediately.
    Can resume from a given start_batch index.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    total_batches = len(texts) // batch_size + int(len(texts) % batch_size != 0)
    if end_batch == -1:
        end_batch = total_batches

    for batch_num in tqdm(range(start_batch, end_batch), total=total_batches): 
        batch_idx = batch_num * batch_size  

        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_df = user_df.iloc[batch_idx:batch_idx + batch_size]

        # Skip empty final batch
        if not batch_texts:
            continue

        # Tokenize the batch
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

        result = []
        for i, emb in enumerate(cls_embeddings):
            row = batch_df.iloc[i]
            result.append({
                'screen_name': row.author,
                'id': row.id,
                'created_utc': row.created_utc,
                'body': row.body,
                'action': row.action,
                'embedding': emb,
            })

        df_batch = pd.DataFrame(result)

        save_path = os.path.join(save_dir, f"batch_{batch_num:04d}.pkl")
        df_batch.to_pickle(save_path)


def process_df(df, tokenizer, model, batch_size=128, save_dir="/modernbert_embed/", start_batch=0, end_batch=-1  ):
    """
    Process a single user's posts to generate batched embeddings and save each batch immediately.
    """
    texts = df.body.tolist()
    texts = [text for text in texts if text.strip()]  # Filter empty strings
    
    get_bert_embedding_batch(texts, df, tokenizer, model, batch_size=batch_size, save_dir=save_dir, start_batch=start_batch, end_batch=end_batch)



sampled_matched_perturbed_df = pd.read_pickle('sampled_matched_perturbed_df.pkl')
user_infos = pd.read_pickle('content_text_df.pkl')

df = pd.DataFrame(user_infos)
df = df.dropna(subset=['body'])
df_sampled_users = df[df.author.isin(sampled_matched_perturbed_df.user.unique())].copy()

print("df loaded")
tokenizer, model = init_worker()

starting_batch = 0
end_batch = -1

process_df(df_sampled_users, tokenizer, model, batch_size=2000, save_dir="modernbert_embed/", start_batch=starting_batch, end_batch=end_batch)

