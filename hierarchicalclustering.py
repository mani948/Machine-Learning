import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# sample data: [Annual Income, Spending Score]
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [20, 76], [21, 6], [22, 94], [23, 72], [24, 60]
])

# perform hierarchical clustering
Z = linkage(X, method='ward')  # 'ward' minimizes variance within clusters

# plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.show()

# cut the dendrogram into 3 clusters
clusters = fcluster(Z, 3, criterion='maxclust')
print("Cluster assignments:", clusters)

# visualize clusters
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='rainbow')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering Example")
plt.show()