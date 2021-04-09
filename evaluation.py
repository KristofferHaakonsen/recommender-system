def evaluate(pred, actual, k_limit=None):
    """
    Evaluate recommendations according to recall@k, ARHR@k, and CTR@k
    k_limit: limit number of articles to recommend. None=no limit
    num_request: The number of requests for predictions
    """

    num_users = len(actual)
    num_requests = 0
    tp_global = 0.
    arhr = 0.
    recalls = []

    # element represents a user and contains it's recommended articles
    for element in pred:
        pred_articles = element.get("articles").tolist()
        if k_limit:
            pred_articles = pred_articles[:k_limit]
        actual_series = actual.loc[element["user_id"]]
        # test_set_positives_list is a list of all test-set entries with value = 1.0 for this user
        test_set_positives_list = list(actual_series[actual_series == 1].index)
        tp_user = 0
        for ind, pred in enumerate(pred_articles):
            num_requests += 1
            if pred in test_set_positives_list:
                # the prediction/suggestion is a true positive (tp)
                tp_global += 1.
                tp_user += 1
                arhr += 1. / float(ind + 1.)
        user_recall = tp_user / len(test_set_positives_list)
        recalls.append(user_recall)

    recall_global = sum(recalls) / num_users
    arhr = arhr / num_users
    ctr = tp_global / num_requests * 100

    prefix = '@'
    if not k_limit:
        k_limit = ''
        prefix = ''
    print("\nGlobal recall{} is {:.4f}".format(
        prefix + str(k_limit), recall_global))
    print("ARHR{} is {:.4f}".format(prefix + str(k_limit), arhr))
    print("CTR{} is {:.4f}%".format(prefix + str(k_limit), ctr))
