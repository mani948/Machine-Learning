import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Use a subset of features for clarity
X_sub = X[:, :3]  # first 3 features
X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.3, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

# 5. Define a simple covariance structure (diagonal weights)
sigma = np.diag(np.random.rand(len(y_train)))  # heteroscedastic variance

# 6. Fit GLS
gls_model = sm.GLS(y_train, X_train_sm, sigma=sigma)
gls_results = gls_model.fit()

# 7. Predict
y_pred = gls_results.predict(X_test_sm)
print(gls_results.summary())
