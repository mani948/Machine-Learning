import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# sample dataset: [Square Footage], [Price]
X = np.array([[500], [750], [1000], [1250], [1500], [1750], [2000]])
y = np.array([100, 150, 200, 250, 300, 350, 400])  # prices in thousands

# train SVR model
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=10)
svr.fit(X, y)

# predict
X_test = np.array([[1200], [1600], [2100]])
predictions = svr.predict(X_test)

print("Predicted Prices:", predictions)

# visualize
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, svr.predict(X), color="red", label="SVR Fit")
plt.xlabel("Square Footage")
plt.ylabel("Price (in thousands)")
plt.title("Support Vector Regression Example")
plt.legend()
plt.show()