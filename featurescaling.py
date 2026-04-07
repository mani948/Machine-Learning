from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Models without scaling
lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)

print("Without Scaling:")
print("Logistic Regression:", accuracy_score(y_test, lr.predict(X_test)))
print("KNN:", accuracy_score(y_test, knn.predict(X_test)))
print("SVM:", accuracy_score(y_test, svm.predict(X_test)))

# 4. Apply scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Models with scaling
lr_s = LogisticRegression(max_iter=200).fit(X_train_scaled, y_train)
knn_s = KNeighborsClassifier().fit(X_train_scaled, y_train)
svm_s = SVC().fit(X_train_scaled, y_train)

print("\nWith Scaling:")
print("Logistic Regression:", accuracy_score(y_test, lr_s.predict(X_test_scaled)))
print("KNN:", accuracy_score(y_test, knn_s.predict(X_test_scaled)))
print("SVM:", accuracy_score(y_test, svm_s.predict(X_test_scaled)))