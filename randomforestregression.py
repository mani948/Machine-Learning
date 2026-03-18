import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# sample data: [Years of Experience]
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000])  # salaries

# create and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# make a prediction
prediction = model.predict([[5.5]])
print("Predicted salary for 5.5 years experience:", prediction)

# visualize
plt.scatter(X, y, color="blue", label="Training data")
X_test = np.linspace(1, 8, 100).reshape(-1, 1)
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred, color="red", label="Random Forest Regression")
plt.scatter(5.5, prediction, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Random Forest Regression Example")
plt.legend()
plt.show()