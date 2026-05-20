from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# 3. Predict cluster labels
y_pred = gmm.predict(X)

# 4. Evaluate clustering performance
print("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
