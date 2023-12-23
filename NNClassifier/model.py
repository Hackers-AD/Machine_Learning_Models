import numpy as np
from datasets import make_classification

class NeuralNetwork:
    def __init__(self, input_units=1, output_units=1, 
                 learning_rate=0.01, hidden_units=64, 
                 hidden_activation='relu') -> None:
        self.lr = learning_rate
        self.hidden_units = hidden_units
        self.nclass = output_units
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.input_weights = np.random.randn(input_units, hidden_units)
        self.hidden_weights = np.random.randn(hidden_units, output_units)
        self.input_bias = np.zeros(hidden_units)
        self.hidden_bias = np.zeros(output_units)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def relu(self, z):
        return np.maximum(0, z)

    def forward_propagation(self, X):
        # Input layer to hidden layer
        self.hidden_input = np.dot(X, self.input_weights) + self.input_bias
        self.hidden_output = self.relu(self.hidden_input)

        # Hidden layer to output layer
        self.output_input = np.dot(self.hidden_output, self.hidden_weights) + self.hidden_bias
        self.output_output = self.softmax(self.output_input)

    def backward_propagation(self, X, y):
        # Calculate gradients for output layer
        output_error = self.output_output - y
        hidden_output_error = np.dot(output_error, self.hidden_weights.T)
        hidden_input_error = hidden_output_error * (self.hidden_input > 0)

        # Update weights and biases
        self.hidden_weights -= self.lr * np.dot(self.hidden_output.T, output_error)
        self.hidden_bias -= self.lr * np.sum(output_error, axis=0, keepdims=True)
        self.input_weights -= self.lr * np.dot(X.T, hidden_input_error)
        self.input_bias -= self.lr * np.sum(hidden_input_error, axis=0)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(X, y)

    def predict(self, X):
        self.forward_propagation(X)
        return np.argmax(self.output_output, axis=1)

if __name__ == "__main__":
    nn = NeuralNetwork(input_units=3, output_units=2, hidden_units=64, learning_rate=0.01)
    features, classes = make_classification(n_samples=100, n_features=3, n_classes=1)
    one_hot_classes = np.eye(2)[classes]
    nn.fit(features, one_hot_classes, epochs=100)

    predictions = nn.predict(features)
    print("Predictions:", predictions)