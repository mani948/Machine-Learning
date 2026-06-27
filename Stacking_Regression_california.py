from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define base learners
base_learners = [
    ('lr', LinearRegression()),
    ('dt', DecisionTreeRegressor(max_depth=6)),
    ('svr', SVR())
]

# 4. Define meta learner
meta_learner = LinearRegression()

# 5. Train Stacking Regression
model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
model.fit(X_train, y_train)

# 6. Predict & Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
