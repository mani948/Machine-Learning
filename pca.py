import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# sample data: [Study Hours, Sleep Hours, Extracurricular Hours]
X = np.array([
    [5, 7, 2],
    [6, 6, 3],
    [7, 5, 4],
    [8, 4, 5],
    [4, 8, 1],
    [3, 9, 2]
])

# apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Reduced Data:\n", X_reduced)

# visualize reduced data
plt.scatter(X_reduced[:,0], X_reduced[:,1], color="blue")
for i, point in enumerate(X_reduced):
    plt.text(point[0]+0.05, point[1]+0.05, f"Student {i+1}")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Dimensionality Reduction Example")
plt.show()