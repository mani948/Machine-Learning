from sklearn.datasets import load_iris
from sklearn.svm import SVR

# 1. Load dataset
iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)   
y = iris.data[:, 3]                  

# 2. Train SVR
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X, y)

# 3. Predict for a sample value
sample = [[1.5]]
print("Predicted petal width for length 1.5:", model.predict(sample)[0])
