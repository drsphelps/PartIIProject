def calc_metrics(classifications, groundtruth):
    metrics = []
    for category in range(max(classifications) + 1):
        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.
        for element in range(len(classifications)):
            if groundtruth[element] == classifications[element]:
                if groundtruth[element] == category:
                    tp += 1.
                else:
                    tn += 1.
            else:
                if classifications[element] == 0:
                    fp += 1.
                else:
                    fn += 1.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2. / ((1./precision) + (1./recall))
        metrics.append({"Precision": precision, "Recall": recall, "F1": f1})

    return metrics
