import numpy as np


def make_regression(n_samples=100, n_features=5, n_targets=1, coef=False, random_state=42):
    features = np.random.uniform(0, 1, (n_samples, n_features)).round(3)
    coeficients = np.random.uniform(0, 1, (n_features, ))
    intercept = 1.5
    target = np.dot(features, coeficients) + intercept

    if coef:
        return features, (target, coeficients)
    
    return features, target


def make_classification(n_samples, n_features=3, n_classes=3, weights=[1, 1, 1], random_state=42):
    sample_size_per_class = n_samples // n_classes
    features = []
    classes = []
    for i in range(n_classes):
        cluster_id = np.full(sample_size_per_class, i+1)
        datapoints = np.random.uniform(0, 1, (sample_size_per_class, n_features)) + 5 * (i + 1)
        for index, id in enumerate(cluster_id):
            classes.append(id)
            features.append(datapoints[index])
    return np.array(features), np.array(classes)



if __name__ == "__main__":
    features, target = make_regression(n_samples=100, n_features=3)
    features, classes = make_classification(n_samples=100, n_features=4)
    print(features.shape, classes.shape)
    
