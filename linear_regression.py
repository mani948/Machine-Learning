import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# sample data
x = np.array([[1], [2], [3], [5], [6]])
y = np.array([2, 4, 6, 7, 8])

# create and train model
model = LinearRegression()
model.fit(x, y)

# make a prediction
prediction = model.predict([[6]])
print("Prediction for input 6:", prediction)

# plot the data points
plt.scatter(x, y, color="blue", label="Data points")

# plot the regression line
plt.plot(x, model.predict(x), color="red", label="Regression line")

# highlight the prediction
plt.scatter(6, prediction, color="green", marker="x", s=100, label="Prediction")

# labels and legend
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression Visualization")
plt.legend()
plt.show()