import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# sample data: [Age, Income]
X = np.array([
    [25, 20000],   # young, low income
    [30, 40000],   # young, medium income
    [45, 80000],   # middle-aged, high income
    [50, 60000],   # middle-aged, medium income
    [65, 70000],   # senior, high income
])
y = np.array([0, 0, 1, 1, 1])  # 0 = No Buy, 1 = Buy

# create and train model
model = DecisionTreeClassifier()
model.fit(X, y)

# make a prediction
prediction = model.predict([[40, 50000]])
print("Prediction for Age=40, Income=50000:", prediction)