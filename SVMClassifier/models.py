import numpy as np
from datasets import make_classification
from model_selection import train_test_split
from metrics import accuracy_score

class SVMClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000, C=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(num_samples):
                condition = y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1

                if condition:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)


if __name__ == "__main__":
    features, targets = make_classification(n_samples=1000, n_features=3, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8)

    svm_classifier = SVMClassifier(learning_rate=0.01, epochs=1000, C=1.0)
    svm_classifier.fit(X_train, y_train)

    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)
