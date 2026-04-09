import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Apply PCA (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Run K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 4. Visualize clusters
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='rainbow')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on Iris (PCA 2D)")
plt.show()

# 5. Compare with true labels
print("Adjusted Rand Index (ARI):", adjusted_rand_score(y, clusters))