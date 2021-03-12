def evaluate(pred, actual, k, num_requests=1, mse = 0.5):
    """
    Evaluate recommendations according to recall@k and ARHR@k
    k: top-k
    """
    # TODO: Add num_requests
    # TODO: Add mse
    total_num = len(actual)
    tp = 0.
    arhr = 0.
    for p, t in zip(pred, actual):
        if t in p:
            tp += 1.
            arhr += 1. / float(p.index(t) + 1.)
    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    ctr = tp / num_requests * 100
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))
    print("CTR@{} is {}%".format(k, ctr))
    print("MSE@{} is {}%".format(k, mse))
