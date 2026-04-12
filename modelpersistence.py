import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Train model
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 2. Save model
joblib.dump(model, "iris_model.pkl")

# 3. Load model
loaded_model = joblib.load("iris_model.pkl")

# 4. Predict
print("Predictions:", loaded_model.predict(X_test[:5]))