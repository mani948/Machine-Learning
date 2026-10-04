from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pac.fit(X_train_scaled, y_train)

# 5. Predict & Evaluate
y_pred = pac.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
