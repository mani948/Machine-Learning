import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Add intercept for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

# 5. Fit Quantile Regression (median, tau=0.5)
model = sm.QuantReg(y_train, X_train_sm)
res = model.fit(q=0.5)

# 6. Predict
y_pred = res.predict(X_test_sm)

# 7. Show results
print("First 5 predictions:", y_pred[:5])
print("Model summary:\n", res.summary())
