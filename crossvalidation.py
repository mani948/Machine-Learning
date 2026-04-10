from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

# 3. Cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
    print(name, "Average Accuracy:", scores.mean())