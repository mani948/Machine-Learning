from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred = agg.fit_predict(X)

# 3. Evaluate clustering performance
print("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
