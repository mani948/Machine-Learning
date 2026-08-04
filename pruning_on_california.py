from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Decision Tree with pruning
dt = DecisionTreeRegressor(random_state=42, ccp_alpha=0.01)  # ccp_alpha controls pruning
dt.fit(X_train_scaled, y_train)

# 5. Predict & Evaluate
y_pred = dt.predict(X_test_scaled)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
