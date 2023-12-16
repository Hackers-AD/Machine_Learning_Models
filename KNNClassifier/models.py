import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def compute_distance(self, X_train, X_test):
        result = []
        for x_sample in X_test:
            x_diff = [np.sqrt(np.sum((x - x_sample) ** 2)) for x in X_train]
            result.append(np.array(x_diff))
        return np.array(result)

    def predict(self, X):
        distance = self.compute_distance(self.X_train, X)
        y_preds = []
        for d in distance:
            lowest_distance_indices = np.argsort(d)[:self.k]
            y_labels = self.y_train[lowest_distance_indices]
            label_predicted = Counter(y_labels).most_common(1)[0][0]
            y_preds.append(label_predicted)
        return np.array(y_preds)