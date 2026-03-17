import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# sample data: [Study Hours, Sleep Hours]
X = np.array([
    [2, 8], [3, 7], [4, 6],   # Fail
    [6, 7], [7, 6], [8, 5]    # Pass
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Fail, 1 = Pass

# create and train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# make a prediction
prediction = model.predict([[5, 6]])
print("Prediction for Study=5, Sleep=6:", "Pass" if prediction[0] == 1 else "Fail")

# visualize data points
plt.scatter(X[y==0, 0], X[y==0, 1], color="red", label="Fail")
plt.scatter(X[y==1, 0], X[y==1, 1], color="blue", label="Pass")
plt.scatter(5, 6, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("Study Hours")
plt.ylabel("Sleep Hours")
plt.title("Random Forest Classification Example")
plt.legend()
plt.show()