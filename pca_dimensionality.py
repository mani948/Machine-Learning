
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load built-in breast cancer dataset
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("Original dataset shape:", X.shape)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce 30 features to 2
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame
pca_data = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1",
             "Principal Component 2"]
)

pca_data["target"] = y

print("\nReduced dataset:")
print(pca_data.head())

print("\nReduced dataset shape:", X_pca.shape)

# Calculate explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal information retained:",
    pca.explained_variance_ratio_.sum() * 100,
    "%"
)

# Visualize the PCA results
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Breast Cancer Dataset")
plt.colorbar(label="Target Class")

plt.show()
