import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# sample data: [Annual Income, Spending Score]
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [20, 76], [21, 6], [22, 94], [23, 72], [24, 60],
    [50, 20], [55, 80], [60, 10], [65, 85], [70, 30]
])

# apply DBSCAN clustering
dbscan = DBSCAN(eps=10, min_samples=2)
clusters = dbscan.fit_predict(X)

print("Cluster assignments:", clusters)

# visualize clusters
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='rainbow')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("DBSCAN Clustering Example")
plt.show()