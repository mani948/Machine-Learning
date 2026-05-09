from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Load dataset
iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)   # Petal length
y = iris.data[:, 3]                  # Petal width

# 2. Transform features into polynomial (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 3. Train Polynomial Regression
model = LinearRegression()
model.fit(X_poly, y)

# 4. Predict for a sample value
sample = poly.transform([[1.5]])
print("Predicted petal width for length 1.5:", model.predict(sample)[0])
