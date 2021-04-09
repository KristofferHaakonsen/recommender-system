def evaluate(pred, actual, k, num_requests):
    """
    Evaluate recommendations according to recall@k, ARHR@k, and CTR@k
    k: top-k
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
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))
    print("CTR@{} is {}%".format(k, ctr))



