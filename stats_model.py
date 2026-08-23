import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# 4. Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

# 5. Fit Quantile Regression (median, tau=0.5)
model = sm.QuantReg(y_train, X_train_sm)
res = model.fit(q=0.5)

# 6. Predict & Evaluate
y_pred = res.predict(X_test_sm)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print(res.summary())
