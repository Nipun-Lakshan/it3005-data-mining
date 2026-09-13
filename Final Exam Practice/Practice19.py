# Note 19: PCA
# ============

# Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Select features
X = df[["Feature1", "Feature2", "Feature3", "Feature4"]]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA Data:")
print(X_pca)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("PCA Components : ", pca.components_)

print("Total Variance:")
print(pca.explained_variance_ratio_.sum())

# Find Number of Components
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print("Number of components:",
      pca.n_components_)
print("Explained variance:",
      pca.explained_variance_ratio_.sum())

# PCA Visualization
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA")
plt.show()

