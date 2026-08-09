from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Use a single feature for isotonic regression (works best in 1D)
X_single = X[:, 0]  # e.g., median income
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.3, random_state=42)

# 3. Train Isotonic Regression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(X_train, y_train)

# 4. Predict & Evaluate
y_pred = iso_reg.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
