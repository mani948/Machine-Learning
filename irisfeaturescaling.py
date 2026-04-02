# Day 15 - Feature Scaling using StandardScaler with KNN

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Step 2: Load the dataset
iris = load_iris()

# Features (input data)
X = iris.data

# Target (flower type)
y = iris.target


# Step 3: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Apply Feature Scaling
scaler = StandardScaler()

# Fit and transform training data
X_train = scaler.fit_transform(X_train)

# Only transform testing data
X_test = scaler.transform(X_test)


# Step 5: Create KNN model
model = KNeighborsClassifier(n_neighbors=5)


# Step 6: Train the model
model.fit(X_train, y_train)


# Step 7: Make predictions
y_pred = model.predict(X_test)


# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Step 9: Print some predictions
print("\nFirst 10 Predictions:", y_pred[:10])


# Step 10: Print actual values
print("\nActual Values:", y_test[:10])