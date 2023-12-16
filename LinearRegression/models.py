import numpy as np

# y = wx + b --> equation of line
class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=100):
        self.lr = learning_rate
        self.num_iters = num_iterations
        self.weight = 0
        self.bias = 0

    def fit(self, X_train, y_train):
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)
        num_samples, num_features = X_train.shape
        self.weight = np.zeros(num_features)
        self.bias = 0

        for i in np.arange(num_samples):
            x_i, y_i = X_train[i], y_train[i]
            y_p = np.dot(x_i, self.weight) + self.bias
            dw = (2/num_samples) * np.dot(x_i.T, (y_p - y_i))
            db = (2/num_samples) * np.sum((y_p - y_i))
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self, X_test):
        y = np.dot(X_test, self.weight) + self.bias
        return y