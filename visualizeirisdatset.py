# Step 1: Import libraries
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# Step 2: Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# Step 3: Select features
# Petal length and Petal width
x_axis = X[:, 2]
y_axis = X[:, 3]


# Step 4: Create scatter plot
plt.scatter(x_axis, y_axis, c=y)


# Step 5: Add labels
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Dataset Visualization")


# Step 6: Show plot
plt.show()