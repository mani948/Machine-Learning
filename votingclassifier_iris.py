from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define base models
log_clf = LogisticRegression(max_iter=200)
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
svm_clf = SVC(kernel='linear', probability=True)

# 4. Voting Classifier (soft voting)
model = VotingClassifier(estimators=[
    ('lr', log_clf), ('dt', tree_clf), ('svc', svm_clf)
], voting='soft')

model.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
