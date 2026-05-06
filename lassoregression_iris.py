from sklearn.datasets import load_iris
from sklearn.linear_model import Lasso

# 1. Load dataset
iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)   # Petal length
y = iris.data[:, 3]                  # Petal width

# 2. Train Lasso Regression
model = Lasso(alpha=0.1)
model.fit(X, y)

# 3. Predict for a sample value
sample = [[1.5]]
print("Predicted petal width for length 1.5:", model.predict(sample)[0])
