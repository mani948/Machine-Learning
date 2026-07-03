import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)

# 4. Map clusters to average target values
cluster_means = {}
for cluster in np.unique(kmeans.labels_):
    cluster_means[cluster] = np.mean(y_train[kmeans.labels_ == cluster])

# 5. Predict by assigning cluster mean
test_clusters = kmeans.predict(X_test)
y_pred = np.array([cluster_means[c] for c in test_clusters])

# 6. Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
