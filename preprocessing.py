import pandas as pd
import os
import json


def clean_data(path='data/active1000', num_days=None):
    """
        Load active1000 dataset and remove events without a document ID.
        @num_days <None|int>:: Custom arg to limit number of days to load. None = load all
    """
    df = load_data(path, num_days)
    df = df[df['documentId'].notnull()]
    df.loc[:, ("publishtime")] = pd.to_datetime(df.publishtime)
    return df


def load_data(path, num_days=None):
    """
        [From project_example]
        Load events from files and convert to dataframe.
        @num_days <None|int>:: Custom arg to limit number of days to load. None = load all
    """
    map_lst = []
    count = 0
    for f in os.listdir(path):
        file_name = os.path.join(path, f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if not obj is None:
                    map_lst.append(obj)
        count += 1
        if num_days and num_days == count:
            break
    return pd.DataFrame(map_lst)
