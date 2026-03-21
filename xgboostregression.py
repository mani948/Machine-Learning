import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# sample data: [Apartment Size in sq ft]
X = np.array([[500], [800], [1000], [1200], [1500], [1800], [2000]])
y = np.array([8000, 12000, 15000, 18000, 22000, 25000, 28000])  # rent in INR

# create and train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# make a prediction
prediction = model.predict([[1300]])
print("Predicted rent for 1300 sq ft apartment:", prediction)

# visualize
plt.scatter(X, y, color="blue", label="Training data")
X_test = np.linspace(500, 2000, 100).reshape(-1, 1)
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred, color="red", label="XGBoost Regression")
plt.scatter(1300, prediction, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("Apartment Size (sq ft)")
plt.ylabel("Rent (INR)")
plt.title("XGBoost Regression Example")
plt.legend()
plt.show()