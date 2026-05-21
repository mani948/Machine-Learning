from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Scale features (important for DBSCAN)
X_scaled = StandardScaler().fit_transform(X)

# 3. Train DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)
y_pred = dbscan.fit_predict(X_scaled)

# 4. Evaluate clustering performance
print("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
