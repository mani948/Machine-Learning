import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# sample data: [BMI, Age]
X = np.array([
    [22, 25], [24, 30], [26, 35],   # No Diabetes
    [30, 40], [32, 45], [35, 50]    # Diabetes
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = No Diabetes, 1 = Diabetes

# create and train model
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
model.fit(X, y)

# make a prediction
prediction = model.predict([[28, 38]])
print("Prediction for BMI=28, Age=38:", "Diabetes" if prediction[0] == 1 else "No Diabetes")

# visualize data points
plt.scatter(X[y==0, 0], X[y==0, 1], color="blue", label="No Diabetes")
plt.scatter(X[y==1, 0], X[y==1, 1], color="red", label="Diabetes")
plt.scatter(28, 38, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("BMI")
plt.ylabel("Age")
plt.title("Gradient Boosting Classification Example")
plt.legend()
plt.show()