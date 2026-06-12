from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train LightGBM Regression
model = LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=-1, random_state=42)
model.fit(X_train, y_train)

# 4. Predict & Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
