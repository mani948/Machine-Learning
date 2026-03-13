import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# sample data: [Size, Weight]
X = np.array([
    [5, 50],   # small fruit
    [6, 55],   # small fruit
    [7, 70],   # medium fruit
    [8, 80],   # medium fruit
    [9, 95],   # large fruit
    [10, 100], # large fruit
])
y = np.array([0, 0, 1, 1, 2, 2])  # 0=Small, 1=Medium, 2=Large

# create and train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# make a prediction
prediction = model.predict([[7.5, 75]])
print("Prediction for fruit with size=7.5, weight=75:", prediction)