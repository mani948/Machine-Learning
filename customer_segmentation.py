
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)

data = pd.DataFrame({
    "annual_income": np.random.randint(15, 120, 200),
    "spending_score": np.random.randint(1, 100, 200)
})

# Select input features
X = data[["annual_income", "spending_score"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Train and assign clusters
data["cluster"] = kmeans.fit_predict(X_scaled)

# Display customer groups
print("Customer Segmentation Results:")
print(data.head(10))

print("\nNumber of customers in each cluster:")
print(data["cluster"].value_counts().sort_index())

# Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    data["annual_income"],
    data["spending_score"],
    c=data["cluster"],
    cmap="viridis"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()

# Predict the cluster for a new customer
new_customer = pd.DataFrame({
    "annual_income": [80],
    "spending_score": [75]
})

new_scaled = scaler.transform(new_customer)

cluster = kmeans.predict(new_scaled)

print("\nNew customer belongs to Cluster:", cluster[0])
