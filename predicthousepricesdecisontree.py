import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# sample data: [Size in sq ft]
X = np.array([[500], [1000], [1500], [2000], [2500]])
y = np.array([100000, 200000, 300000, 400000, 500000])  # prices

# create and train model
model = DecisionTreeRegressor()
model.fit(X, y)

# make a prediction
prediction = model.predict([[1800]])
print("Predicted price for 1800 sq ft house:", prediction)

# visualize
plt.scatter(X, y, color="blue", label="Training data")
plt.plot(X, model.predict(X), color="red", label="Regression fit")
plt.scatter(1800, prediction, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.title("Decision Tree Regression Example")
plt.legend()
plt.show()