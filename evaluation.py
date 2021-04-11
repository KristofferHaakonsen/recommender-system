def evaluate(pred, actual, k_limit=None):
    """
    Evaluate recommendations according to recall@k, ARHR@k, and CTR@k
    k_limit: limit number of articles to recommend. None=no limit
    """

    num_users = len(actual)     
    tp_global = 0.
    arhr = 0.
    num_actual = 0      # The numbers of actual articles

    # element represents a user and contains it's recommended articles
    for element in pred:
        pred_articles = element.get("articles").tolist()
        if k_limit:
            pred_articles = pred_articles[:k_limit]
        actual_series = actual.loc[element["user_id"]]
        # test_set_positives_list is a list of all test-set entries with value = 1.0 for this user
        test_set_positives_list = list(actual_series[actual_series == 1].index)
        num_actual += len(test_set_positives_list) # Count all the actuall clicks
        for ind, pred in enumerate(pred_articles):
            if pred in test_set_positives_list:
                # the prediction/suggestion is a true positive (tp)
                tp_global += 1.
                arhr += 1. / float(ind + 1.)


    arhr = arhr / num_users
    ctr = tp_global / num_actual * 100
    recall = tp_global/num_actual      

    prefix = '@'
    if not k_limit:
        k_limit = ''
        prefix = ''
    print("\nRecall{} is {:.4f}".format(prefix + str(k_limit), recall))
    print("ARHR{} is {:.4f}".format(prefix + str(k_limit), arhr))
    print("CTR{} is {:.4f}%".format(prefix + str(k_limit), ctr))
