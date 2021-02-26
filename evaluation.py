def evaluate(pred, actual, k):
    """
    Evaluate recommendations according to recall@k and ARHR@k
    k: top-k
    """
    total_num = len(actual)
    tp = 0.
    arhr = 0.
    for p, t in zip(pred, actual):
        if t in p:
            tp += 1.
            arhr += 1./float(p.index(t) + 1.)
    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))

    # TODO get each metric
    # TODO return all needed metrics
