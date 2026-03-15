import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# sample data: [x, y]
X = np.array([
    [1, 2], [2, 3], [3, 3],   # Class 0
    [6, 6], [7, 7], [8, 6]    # Class 1
])
y = np.array([0, 0, 0, 1, 1, 1])  # labels

# create and train SVM model
model = svm.SVC(kernel='linear')
model.fit(X, y)

# make a prediction
prediction = model.predict([[4, 4]])
print("Prediction for point (4,4):", prediction)

# visualize
plt.scatter(X[y==0, 0], X[y==0, 1], color="blue", label="Class 0")
plt.scatter(X[y==1, 0], X[y==1, 1], color="red", label="Class 1")
plt.scatter(4, 4, color="green", marker="x", s=100, label="Prediction")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("SVM Classification Example")
plt.legend()
plt.show()