import json
import pandas as pd
import os
import numpy as np


def load_content_data(path):
    article_lst = []
    extract_fields = ['id', 'publishtime', 'title', 'description', 'keyword',
                      'kw-classification', 'kw-category', 'kw-concept', 'kw-company', 'kw-entity', 'kw-location']
    for f in os.listdir(path):
        file_name = os.path.join(path, f)
        if os.path.isfile(file_name):
            obj = json.loads(open(file_name).readline().strip())
            if not obj is None:
                d = {}
                for field in obj['fields']:
                    if field['field'] in extract_fields:
                        d[field['field']] = field['value']
                article_lst.append(d)
    df = pd.DataFrame(article_lst)
    df.loc[:, ("publishtime")] = pd.to_datetime(df.publishtime)
    return df


def load_events_data(path='data/active1000', num_days=None):
    """
        Load active1000 dataset and remove events without a document ID.
        @num_days <None|int>:: Custom arg to limit number of days to load. None = load all
    """
    df = read_event_data(path, num_days)
    df = df[df['documentId'].notnull()]
    df.loc[:, ("publishtime")] = pd.to_datetime(df.publishtime)
    return df


def read_event_data(path, num_days=None):
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


def merge_on_docID(left, right):
    df = pd.merge(left, right, how='inner',
                  on=['documentId'])
    return df


def list_str_concat(x):
    if not x:
        return []
    elif isinstance(x, list):
        return x
    return [x]


def keyword_freq(keywords):
    freqs = {}
    for keyword in keywords:
        # some articles were missing keywords or category
        if pd.isnull(keyword):
            continue
        elif not isinstance(keyword, str):
            kw = keyword
        else:
            kw = keyword.lower()

        if kw in freqs:
            freqs[kw] += 1
        else:
            freqs[kw] = 1
    return freqs


def get_user_counts(df_merged):
    '''
        returns: df of frequencies (keyword and category) for each user
    '''
    df_merged['keyword'] = df_merged['keyword'].apply(list_str_concat)
    df_merged['category'] = df_merged['category'].apply(list_str_concat)
    merged_kw = df_merged.groupby("userId").agg(
        {'keyword': 'sum', 'category': 'sum'})
    merged_kw['keyword'] = merged_kw['keyword'].apply(keyword_freq)
    merged_kw['category'] = merged_kw['category'].apply(keyword_freq)
    return merged_kw


def combine_dicts_in_row(row):
    return {**row['category'], **row['keyword']}


def get_user_profile(df):
    def prefix_cat(dic):
        new_dict = {}
        for key in dic:
            new_dict['CAT_' + key] = dic[key]
        return new_dict

    def weight_multiply(num):
        return 4 * num

    def weight_divide(num):
        return 0.25 * num

    df['category'] = df['category'].apply(prefix_cat)
    df['combined_keywords'] = df.apply(combine_dicts_in_row, axis=1)

    user_pref_df = pd.DataFrame.from_records(
        df['combined_keywords'], index=df.index)

    # convert NaN to 0
    user_pref_df = user_pref_df.fillna(0)

    # apply weights (decrease categories, increase keywords)
    for column_name in user_pref_df.columns:
        if column_name[0:4] == 'CAT_':
            user_pref_df[column_name] = user_pref_df[column_name].apply(
                weight_divide)
        else:
            user_pref_df[column_name] = user_pref_df[column_name].apply(
                weight_multiply)

    return user_pref_df
