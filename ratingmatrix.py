import pandas as pd
import json
import numpy as np


def rating_matrix(df, debugSlice=None):
    '''
        [Much of this code is from project_example]
        return a simple rating_matrix where rating is activeTime.
        @debugSlice <None|int>: if int, use only {debugSlice} first rows in df 
    '''

    # TODO keep this? in case the df includes nan activeTime
    df['activeTime'] = df['activeTime'].fillna(0)
    df = df[~df['documentId'].isnull()]
    # TODO: we might be dropping valuable info below..
    df = df.drop_duplicates(subset=['userId', 'documentId'])
    df = df.sort_values(by=['userId', 'time'])

    # Slice the RM for debug purposes
    if debugSlice:
        df = df[:debugSlice]

    n_users = df['userId'].nunique()
    n_items = df['documentId'].nunique()

    ratings = np.zeros((n_users, n_items))

    # create column 'uid' which is a simplification of userId and in order
    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    new_user = np.r_[True, new_user]
    df['uid'] = np.cumsum(new_user)

    # create a simplified documentId "tid" and merge this with the original df
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame(
        {'documentId': item_ids, 'tid': range(1, len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_ext = df[['uid', 'tid', 'activeTime']]

    # add 1 as rating where there exists an event
    for row in df_ext.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    return ratings


def rating_matrix_train_test_split(rm, fraction=0.2):
    """
        [From project_example]
        Leave out a fraction of dataset for test use
    """
    test = np.zeros(rm.shape)
    train = rm.copy()
    for user in range(rm.shape[0]):
        size = int(len(rm[user, :].nonzero()[0]) * fraction)
        # indexes or ("item-number j") to use for testing
        test_ratings = np.random.choice(rm[user, :].nonzero()[0],
                                        size=size,
                                        replace=False)
        # set the test_ratings to 0 in train
        train[user, test_ratings] = 0.
        test[user, test_ratings] = rm[user, test_ratings]
    return train, test


if __name__ == '__main__':
    # load temp_data
    map_lst = []
    for line in open('data/active1000/20170101'):
        obj = json.loads(line.strip())
        if obj is not None:
            map_lst.append(obj)
    df = pd.DataFrame(map_lst)

    rm = rating_matrix(df, debugSlice=200)
    train, test = rating_matrix_train_test_split(rm)
