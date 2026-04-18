from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# 1. Load dataset
iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)   # Petal length
y = iris.data[:, 3]                  # Petal width

# 2. Train Linear Regression
model = LinearRegression()
model.fit(X, y)

# 3. Print coefficients
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
