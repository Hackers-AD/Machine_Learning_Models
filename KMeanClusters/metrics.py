import numpy as np

def predict_accuracy(y_true, y_predicted):
    hits = 0
    miss = 0
    for idx in np.arange(len(y_true)):
        if y_true[idx] == y_predicted[idx]:
            hits += 1
        else:
            miss += 1
    accuracy = np.round(hits/(hits + miss), 10)
    return accuracy