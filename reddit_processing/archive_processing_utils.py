
import orjson
import pandas as pd


def load_df_jsonl(path):
    with open(path, 'rb') as f:
        data = [orjson.loads(line) for line in f]
    return pd.DataFrame(data)