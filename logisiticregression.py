import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([[1],[2],[3],[5],[6]])
y = np.array([2,4,6,7,8])

model = LinearRegression()
model.fit(x, y)

prediction = model.predict([[6]])

plt.scatter(x, y, color="blue")              # data points
plt.plot(x, model.predict(x), color="red")   # regression line
plt.scatter(6, prediction, color="green")    # prediction point
plt.show()