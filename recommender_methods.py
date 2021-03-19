
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from evaluation import evaluate
from example_code import project_example
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import average, exp


def collaborative_filtering(train, test):
    """
    [From project_example, slightly edited]
    performes collaborative_filtering on train, test and plots the learning curve.
    """
    # train and test model with matrix factorization
    mf_als = project_example.mf.ExplicitMF(train, n_factors=40,
                                           user_reg=0.0, item_reg=0.0)
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    mf_als.calculate_learning_curve(iter_array, test)
    # plot out learning curves
    plot_learning_curve(iter_array, mf_als)


def plot_learning_curve(iter_array, model):
    """
    [From project_example, fixed faulty code in explicitMF]
    plots learning_curve over iterations
    """
    print(iter_array, model.train_mse)
    plt.plot(iter_array, model.train_mse,
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse,
             label='Test', linewidth=5)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('MSE', fontsize=30)
    plt.legend(loc='best', fontsize=20)
    plt.show()


def content_processing(df):
    """
        [From project_example]
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    df = df[df['documentId'].notnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df['category'] = df['category'].str.split('|')
    df['category'] = df['category'].fillna("").astype('str')

    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame(
        {'documentId': item_ids, 'tid': range(1, len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'category'], ascending=True, inplace=True)

    # select features/words using TF-IDF
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
    tfidf_matrix = tf.fit_transform(df_item['category'])
    print('Dimension of feature vector: {}'.format(tfidf_matrix.shape))
    # measure similarity of two articles with cosine similarity

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    print("Similarity Matrix:")
    print(cosine_sim[:4, :4])
    return cosine_sim, df


def content_recommendation(df, k=20):
    """
        [From project_example]
        Generate top-k list according to cosine similarity
    """
    cosine_sim, df = content_processing(df)
    df = df[['userId', 'time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    print(df[:20])  # see how the dataset looks like
    pred, actual = [], []
    puid, ptid1, ptid2 = None, None, None
    for row in df.itertuples():
        uid, tid = row[1], row[3]
        if uid != puid and puid is not None:
            idx = ptid1
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            sim_scores = [i for i, j in sim_scores]
            pred.append(sim_scores)
            actual.append(ptid2)
            puid, ptid1, ptid2 = uid, tid, tid
        else:
            ptid1 = ptid2
            ptid2 = tid
            puid = uid
    evaluate(pred, actual, k)
    return pred, actual


def decay_function(target_time, user_item_time):
    decay_rate = 0.8
    return exp(- decay_rate * (target_time - user_item_time))


def collaborative_filtering_user_based(rm, userID, k=2):
    """
    performes collaborative_filtering on rating matrix and plot the learning curve.
    """

    # Compute peer group of the user
    for user in range(len(rm)):
        if user != userID:
            similarity, p_value = pearsonr(rm[userID][1:], rm[user][1:])
            mean = average(rm[user][1:])
            rm[user] = rm[user] + [similarity] + [mean]

    # Add for own user. Because sort fails if it is shorter
    rm[userID] = rm[userID] + [1] + [average(rm[userID][1:])]

    rm.sort(reverse=False, key=lambda x: x[5])

    print("AFTER SORT: ", rm)

    k_closest = rm[1: k + 1]

    print("K CLOSEST ", k_closest)

    # Predict the rate of item j

    '''numerator = 0
    denominator = 0

    for user in range(len(k_closest)):
        # for item in range(len(k_closest[user]) - 2):
        numerator += k_closest[user][5] * (k_closest[user][3] - k_closest[user][6])
        denominator += abs(k_closest[user][5])

    ruj = rm[userID][6] + numerator / denominator
    print(ruj)'''

    return k_closest



# Plot prediction


if __name__ == '__main__':
    # load temp_data
    # map_lst = []
    # for line in open('data/active1000/20170101'):
    # obj = json.loads(line.strip())
    #  if obj is not None:
    #       map_lst.append(obj)
    # df = pd.DataFrame(map_lst)

    # rm = rating_matrix(df, debugSlice=200)

    # content_recommendation(df)

    ### Collaboratib filtering

    rm = [["u0", 10, 1, 3, 0, 1],
          ["u1",30, 20, 4, 10, 1],
          ["u2", 5, 30, 2, 6, 7],
          ["u3", 6, 20, 6, 0, 4],
          ["u4", 10, 20, 5, 0, 1]]

    collaborative_filtering_user_based(rm, 0)
