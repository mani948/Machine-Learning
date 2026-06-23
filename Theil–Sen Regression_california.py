from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale features (important for Theil–Sen)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Theil–Sen Regression (robust to outliers)
model = TheilSenRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test_scaled)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
