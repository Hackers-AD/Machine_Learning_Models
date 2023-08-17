import numpy as np

def train_test_split(X, y=None, train_size=0.25):
    num_train_samples = int(len(X) * train_size)
    indices_array = np.arange(0, len(X))
    indices_selected = np.random.choice(indices_array, num_train_samples, replace=False)
    remaining_indices = np.setdiff1d(indices_array, indices_selected)

    X_train, X_test = X[indices_selected], X[remaining_indices]

    if y is not None:
        y_train, y_test = y[indices_selected], y[remaining_indices]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test