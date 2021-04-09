def evaluate(pred, actual, num_requests, k_limit=None):
    """
    Evaluate recommendations according to recall@k, ARHR@k, and CTR@k
    k_limit: limit number of articles to recommend. None=no limit
    num_request: The number of requests for predictions
    """

    total_num = len(actual)
    tp = 0.
    arhr = 0.

    # element represents a user and contains it's recommended articles
    for element in pred:
        pred_articles = element.get("articles").tolist()
        if k_limit:
            pred_articles = pred_articles[:k_limit]
        for p, t in zip(pred_articles, actual):
            if t in p:
                tp += 1.
                arhr += 1. / float(pred_articles.index(t) + 1.)

    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    ctr = tp / num_requests * 100

    prefix = '@'
    if not k_limit:
        k_limit = ''
        prefix = ''
    print("\nRecall{} is {:.4f}".format(prefix + str(k_limit), recall))
    print("ARHR{} is {:.4f}".format(prefix + str(k_limit), arhr))
    print("CTR{} is {}%".format(prefix + str(k_limit), ctr))
