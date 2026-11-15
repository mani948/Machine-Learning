
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Generate sample transaction dataset
np.random.seed(42)

data = pd.DataFrame({
    "amount": np.random.uniform(10, 1000, 1000),
    "hour": np.random.randint(0, 24, 1000),
    "transactions_per_day": np.random.randint(1, 20, 1000)
})

# Create sample fraud labels
data["fraud"] = (
    (data["amount"] > 800) &
    (data["hour"] < 5)
).astype(int)

# Separate input and output
X = data.drop("fraud", axis=1)
y = data["fraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Predict a new transaction
new_transaction = pd.DataFrame({
    "amount": [950],
    "hour": [2],
    "transactions_per_day": [3]
})

result = model.predict(new_transaction)

if result[0] == 1:
    print("Alert: Potential Fraud Transaction!")
else:
    print("Transaction appears Normal.")
