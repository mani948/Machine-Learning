
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create sample customer dataset
np.random.seed(42)

data = pd.DataFrame({
    "age": np.random.randint(18, 65, 500),
    "tenure": np.random.randint(1, 60, 500),
    "monthly_bill": np.random.randint(20, 150, 500),
    "support_calls": np.random.randint(0, 10, 500)
})

# Create target column
data["churn"] = (
    (data["tenure"] < 12) |
    (data["support_calls"] > 6)
).astype(int)

# Separate input and output
X = data.drop("churn", axis=1)
y = data["churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN model
model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)

# Evaluate model
predictions = (model.predict(X_test) > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Predict for a new customer
new_customer = np.array([[35, 8, 90, 7]])

new_customer = scaler.transform(new_customer)

result = model.predict(new_customer)

if result[0][0] > 0.5:
    print("Customer may leave")
else:
    print("Customer may stay")
