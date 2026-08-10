from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
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

# 4. Try different kernels
kernels = ['linear', 'poly', 'rbf']
for k in kernels:
    kr = KernelRidge(kernel=k, alpha=1.0)
    kr.fit(X_train_scaled, y_train)
    y_pred = kr.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Kernel: {k}, MSE: {mse:.4f}")
