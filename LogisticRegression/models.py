import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
class LogisticRegression:
    def __init__(self, num_iters=1000, learning_rate=0.006):
        self.lr = learning_rate
        self.num_iters = num_iters
        self.weights = None
        self.bias = 0

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.num_iters):
            y_linear = np.dot(X_train, self.weights) + self.bias
            y_pred = self.sigmoid(y_linear)
            dw = (1/n_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1/n_samples) * np.sum(y_pred - y_train)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self, X_test):
        y_linear = np.dot(X_test, self.weights) + self.bias
        y_pred = self.sigmoid(y_linear)
        return np.round(y_pred)

    def accuracy_score(self, y_true, y_predicted):
        hits = 0
        misses = 0
        for idx, y in enumerate(y_true):
            hits = (hits + 1) if y_predicted[idx] == y else hits
            misses = (misses + 1) if y_predicted[idx] != y else misses

        score = hits / (hits + misses)
        return score