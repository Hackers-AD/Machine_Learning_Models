import numpy as np

class KMeanClusters:
    def __init__(self, k=3):
        self.k = k
        self.train_data = None
        self.centroids = None
        self.clusters = None

    def calculate_euclidean_distance(self, X1, X2):
        distance = np.sum((X1 -X2) ** 2)
        return np.sqrt(distance)

    def fit(self, train_data):
        indices_sel = np.random.choice(np.arange(len(train_data)), self.k, replace=False)
        self.centroids = train_data[indices_sel]
        self.clusters = [[] for _ in range(self.k)]

        for data in train_data:
            data_distance = []
            for centroid in self.centroids:
                d = self.calculate_euclidean_distance(data, centroid)
                data_distance.append(d)
            centroid_idx_sel = np.argmin(data_distance)
            self.clusters[centroid_idx_sel].append(data)
            self.centroids[centroid_idx_sel] = np.mean(self.clusters[centroid_idx_sel], axis=0)

    def predict(self, test_data):
        predicted_labels = []
        clusters = [[] for _ in range(self.k)]
        for data in test_data:
            data_distance = []
            for centroid in self.centroids:
                d = self.calculate_euclidean_distance(data, centroid)
                data_distance.append(d)

            centroid_idx_sel = np.argmin(data_distance)
            clusters[centroid_idx_sel].append(data)
            predicted_labels.append(centroid_idx_sel)
        return np.array(predicted_labels), np.array(clusters, dtype=object)