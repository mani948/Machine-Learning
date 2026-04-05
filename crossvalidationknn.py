# Day 18 - Cross Validation using KNN

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Step 2: Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# Step 3: Create model
model = KNeighborsClassifier(n_neighbors=5)


# Step 4: Perform cross validation
scores = cross_val_score(model, X, y, cv=5)


# Step 5: Print results
print("Cross Validation Scores:", scores)


# Step 6: Print average accuracy
print("Average Accuracy:", np.mean(scores))