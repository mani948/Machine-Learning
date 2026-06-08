from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)

# 4. Predict cluster centers as regression proxy
y_pred = np.array([kmeans.cluster_centers_[label][0] for label in kmeans.predict(X_test)])

# 5. Evaluate
print("Mean Squared Error (cluster proxy):", mean_squared_error(y_test, y_pred))
