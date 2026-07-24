from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train Random Forest (used for QRF)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Predict quantiles (e.g., 10th, 50th, 90th percentile)
preds = []
for tree in model.estimators_:
    preds.append(tree.predict(X_test))
preds = np.array(preds)

q10 = np.percentile(preds, 10, axis=0)
q50 = np.percentile(preds, 50, axis=0)  # median
q90 = np.percentile(preds, 90, axis=0)

print("Sample predictions (first 5 rows):")
for i in range(5):
    print(f"10%={q10[i]:.2f}, Median={q50[i]:.2f}, 90%={q90[i]:.2f}")
