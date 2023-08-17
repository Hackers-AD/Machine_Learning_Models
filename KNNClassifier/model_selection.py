import numpy as np

def train_test_split(X, y, train_size=0.25):
    num_train_samples = int(len(X) * train_size)
    indices_array = np.arange(0, len(X))
    indices_selected = np.random.choice(indices_array, num_train_samples, replace=False)
    remaining_indices = np.setdiff1d(indices_array, indices_selected)

    X_train, y_train = X[indices_selected], y[indices_selected]
    X_test, y_test = X[remaining_indices], y[remaining_indices]

    return X_train, X_test, y_train, y_test