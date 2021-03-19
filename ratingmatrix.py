import pandas as pd
import json
import numpy as np


def rating_matrix(df, debugSlice=None):
    '''
        return a df rating_matrix where ratings are 1 if read.
        @debugSlice <None|int>: if int, use only {debugSlice} first rows in df
    '''

    # add 1 as rating where there exists an event
    def add_to_user_reading_list(row):
        if not row['userId'] in reading_lists:
            reading_lists[row['userId']] = {}
        reading_lists[row['userId']][row['documentId']] = 1

    # for each userId, a dictionary of read documentIds
    reading_lists = {}

    df = df.drop_duplicates(subset=['userId', 'documentId'])
    # df = df.sort_values(by=['userId', 'time'])

    # Slice the RM for debug purposes
    if debugSlice:
        df = df[:debugSlice]

    # n_users = df['userId'].nunique()
    # n_items = df['documentId'].nunique()

    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame(
        {'documentId': item_ids})
    df = pd.merge(df, new_df, on='documentId', how='outer')

    df.apply(add_to_user_reading_list, axis=1)

    rating_matrix_df = pd.DataFrame.from_records(
        list(reading_lists.values()), index=reading_lists.keys())

    rating_matrix_df = rating_matrix_df.fillna(0.0)

    return rating_matrix_df


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
