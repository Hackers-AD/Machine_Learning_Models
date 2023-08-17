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

def mse_score(y_true, y_predicted):
    num_samples = y_true.shape[0]
    score = (np.sum((y_true - y_predicted) ** 2))/num_samples
    return score

def r2_score(y_true, y_predicted):
    num_samples = y_true.shape[0]
    mean_true = np.mean(y_true)

    total_sum_squared = np.sum((y_true - mean_true) ** 2)
    residual_sum_squared = np.sum((y_true - y_predicted) ** 2)

    score = 1 - (residual_sum_squared/total_sum_squared)
    return score