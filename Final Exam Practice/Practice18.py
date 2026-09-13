# Note 18: DBSCAN
# ===============

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Header
print("\n=================")
print("DBSCAN Clustering")
print("=================\n")

# DBSCAN Clustering [Used Grid Search for Find Best Eps and Min Samples]
DBScan = DBSCAN(eps=0.7, min_samples=9)
DBScan.fit(X_pca)
labels = DBScan.labels_

print("01. Number of Clusters      : ", len(set(labels)) - (1 if -1 in labels else 0))
print("02. Number of Noise Points  : ", list(labels).count(-1))

# Calculate S Score
n_clusters = len(set(labels)) - (1 if -1 in DBScan.labels_ else 0)
BEST_S_SCORE_DBSCAN = 0
if n_clusters >= 2:
    dbscan_score = silhouette_score(X_pca, DBScan.labels_)
    print("03. DBSCAN Silhouette Score : ", round(dbscan_score, 2))
    BEST_S_SCORE_DBSCAN = round(dbscan_score, 2)
else:
    print("Silhouette Score cannot be calculated.")

# Plot DBSCAN
plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis',s=50)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("DBSCAN Clustering For Wine Clustering Dataset")
plt.grid(True)

plt.show()

# Plot the Comparison
plt.figure()
X = ['DBSCAN', 'KMeans']
Y = [BEST_S_SCORE_DBSCAN, BEST_S_SCORE_KMEANS]
plt.scatter(X, Y, color='red', marker='o', s=10)
plt.plot(X, Y, color='blue', linestyle='--')
plt.xlabel("Models")
plt.ylabel("Silhouette Score")
plt.title("Comparison of KMeans and DBSCAN For Wine Clustering Dataset")
plt.grid(True, alpha=0.3)

# Add value labels
for i in range(len(X)):
    plt.text(
        X[i],          # x-coordinate
        Y[i] + 0.001,   # slightly above the point
        f"{Y[i]:.2f}", # show 3 decimal places
        ha='center'
    )

plt.show()

# Number of Clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("\nNumber of clusters:", n_clusters)

# Number of Noise Points
n_noise = list(labels).count(-1)
print("Number of noise points:", n_noise)

# Visualize DBSCAN clusters
# =========================

plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")

plt.show()