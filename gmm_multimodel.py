import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Use a single feature for simplicity (e.g., median income)
X_single = X[:, [0]]  # median income
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.3, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(np.column_stack([X_train_scaled, y_train]))

# 5. Predict mean of conditional distribution
def gmr_predict(gmm, X_input):
    preds = []
    for x in X_input:
        # Condition on X
        means = gmm.means_
        covs = gmm.covariances_
        weights = gmm.weights_
        # Extract conditional expectation (simplified for 1D X)
        y_mean = np.sum(weights * means[:,1])  # approximate conditional mean
        preds.append(y_mean)
    return np.array(preds)

y_pred = gmr_predict(gmm, X_test_scaled)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
