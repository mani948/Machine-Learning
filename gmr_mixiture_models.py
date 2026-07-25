import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(X_train)

# 4. Map clusters to average target values
train_clusters = gmm.predict(X_train)
cluster_means = {c: np.mean(y_train[train_clusters == c]) for c in np.unique(train_clusters)}

# 5. Predict on test data
test_clusters = gmm.predict(X_test)
y_pred = np.array([cluster_means[c] for c in test_clusters])

# 6. Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
