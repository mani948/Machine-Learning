from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 1. Load dataset
iris = load_iris()
X = iris.data

# 2. Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 3. Print first 5 reduced samples
print("Reduced data (first 5 rows):")
print(X_reduced[:5])
