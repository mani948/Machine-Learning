import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Visualize
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris dataset with PCA (2D)")
plt.show()

# 4. Train classifier on reduced data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy with PCA (2D):", accuracy_score(y_test, y_pred))

# 5. Compare with original features
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X, y, test_size=0.3, random_state=42)
clf_o = LogisticRegression(max_iter=200)
clf_o.fit(X_train_o, y_train_o)
y_pred_o = clf_o.predict(X_test_o)

print("Accuracy with original features:", accuracy_score(y_test_o, y_pred_o))