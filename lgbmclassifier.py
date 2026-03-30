import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Load dataset
df = pd.read_csv("tournament_data.csv")

# Features
X = df[["seed_diff", "win_ratio_diff", "score_margin_diff"]]
y = df["teamA_win"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
probs = model.predict_proba(X_test)[:,1]
print("Log Loss:", log_loss(y_test, probs))