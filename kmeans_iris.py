from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. Load dataset
iris = load_iris()
X = iris.data

# 2. Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. Print cluster labels for first 10 samples
print("Cluster labels:", kmeans.labels_[:10])
