import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset (wine classification)
wine = load_wine()
X, y = wine.data, wine.target

# 2. Binarize labels for PLS (multi-class → one-hot)
lb = LabelBinarizer()
Y = lb.fit_transform(y)

# 3. Split data
X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y, test_size=0.3, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train PLS-DA (using PLSRegression)
pls_da = PLSRegression(n_components=5)
pls_da.fit(X_train_scaled, Y_train)