from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define kernel (RBF with constant scaling)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)

# 4. Train Gaussian Process Regression
model = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, random_state=42)
model.fit(X_train[:2000], y_train[:2000])  # GP is heavy, so use subset

