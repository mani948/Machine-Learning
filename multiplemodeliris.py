# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Step 2: Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Create models
knn_model = KNeighborsClassifier(n_neighbors=5)
dt_model = DecisionTreeClassifier()


# Step 5: Train models
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)


# Step 6: Make predictions
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)


# Step 7: Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)


# Step 8: Print results
print("KNN Accuracy:", knn_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)