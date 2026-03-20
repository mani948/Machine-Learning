import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# sample data: [Age, Income]
X = np.array([
    [22, 20000], [25, 25000], [28, 30000],   # No Purchase
    [35, 50000], [40, 60000], [45, 70000]    # Purchase
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = No Purchase, 1 = Purchase

# create and train model
model = XGBClassifier(n_estimators=50, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# make a prediction
prediction = model.predict([[30, 40000]])

print("Prediction for Age=30, Income=40000:", "Purchase" if prediction[0] == 1 else "No Purchase")

# visualize data points
plt.scatter(X[y==0, 0], X[y==0, 1], color="red", label="No Purchase")
plt.scatter(X[y==1, 0], X[y==1, 1], color="blue", label="Purchase")
plt.scatter(30, 40000, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("XGBoost Classification Example")
plt.legend()
plt.show()