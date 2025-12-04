import pandas as pd
import numpy as np

from datetime import timedelta
from datetime import datetime

def merge_df(df_a,df_b):
    df_total = pd.concat([df_a,df_b])
    df_total = df_total.sort_values(by=['created_utc',"action"]).copy()
    df_total.reset_index(inplace = True)
    del df_total['index']
    return df_total


