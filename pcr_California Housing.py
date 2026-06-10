from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Apply PCA (reduce to 5 components)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. Train Linear Regression on reduced data
model = LinearRegression()
model.fit(X_train_pca, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test_pca)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
