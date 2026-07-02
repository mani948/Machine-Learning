import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Discretize target for Naive Bayes (approximate regression)
y_bins = np.floor(y)   # bucket continuous values into integers

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_bins, test_size=0.3, random_state=42)

# 4. Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Predict & Evaluate (map back to continuous buckets)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
