from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Step 1: Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 3: Create model
model = KNeighborsClassifier(n_neighbors=5)


# Step 4: Train model
model.fit(X_train, y_train)


# Step 5: Make predictions
y_pred = model.predict(X_test)


# Step 6: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)