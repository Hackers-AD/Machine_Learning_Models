import numpy as np
from models import KMeanClusters
from model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    cluster1 = np.random.normal(0, 2, (5000, 10)) + np.random.rand(10) * 10
    cluster2 = np.random.normal(0, 3, (5000, 10)) + np.random.rand(10) * 20
    cluster3 = np.random.normal(0, 1, (5000, 10)) + np.random.rand(10) * 30
    data = np.vstack([cluster1, cluster2, cluster3])

    train_data, test_data = train_test_split(data, train_size=0.8)

    model = KMeanClusters(k=5)
    model.fit(data)

    labels, clusters = model.predict(test_data)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.scatter(test_data[:,0], test_data[:,1], c=labels)
    plt.show()




