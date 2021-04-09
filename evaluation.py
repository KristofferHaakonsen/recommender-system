def evaluate_top_k(pred, actual, num_requests, k):
    """
    Evaluate recommendations according to recall@k, ARHR@k, and CTR@k
    k: top-k
    num_request: The number of requests for predictions
    """

    total_num = len(actual)
    tp = 0.
    arhr = 0.

    # element represents a user and contains it's recommended articles
    for element in pred:
        pred_articles = element.get("articles").tolist()
        pred_articles = pred_articles[:k]
        for p, t in zip(pred_articles, actual):
            if t in p:
                tp += 1.
                arhr += 1. / float(pred_articles.index(t) + 1.)

    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    ctr = tp / num_requests * 100

    print("\nRecall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))
    print("CTR@{} is {}%".format(k, ctr))


def evaluate(pred, actual, num_requests):
    """
    Evaluate recommendations according to recall, ARHR, and CTR
    num_request: The number of requests for predictions
    """

    total_num = len(actual)
    tp = 0.
    arhr = 0.

    for element in pred:
        pred_articles = element.get("articles").tolist()
        for p, t in zip(pred_articles, actual):
            if t in p:
                tp += 1.
                arhr += 1. / float(pred_articles.index(t) + 1.)

    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    ctr = tp / num_requests * 100

    print("\nRecall is {:.4f}".format(recall))
    print("ARHR is {:.4f}".format(arhr))
    print("CTR is {}%".format(ctr))
