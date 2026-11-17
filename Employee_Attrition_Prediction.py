
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Generate sample employee dataset
np.random.seed(42)

data = pd.DataFrame({
    "age": np.random.randint(20, 60, 1000),
    "salary": np.random.randint(15000, 100000, 1000),
    "experience": np.random.randint(0, 20, 1000),
    "satisfaction": np.random.randint(1, 6, 1000),
    "overtime": np.random.randint(0, 2, 1000)
})

# Create sample attrition labels
data["attrition"] = (
    (data["satisfaction"] <= 2) |
    ((data["overtime"] == 1) &
     (data["experience"] < 3))
).astype(int)

# Separate input and output
X = data.drop("attrition", axis=1)
y = data["attrition"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Predict for a new employee
new_employee = pd.DataFrame({
    "age": [25],
    "salary": [25000],
    "experience": [1],
    "satisfaction": [2],
    "overtime": [1]
})

result = model.predict(new_employee)

if result[0] == 1:
    print("\nPrediction: Employee may leave")
else:
    print("\nPrediction: Employee may stay")

# Display feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
