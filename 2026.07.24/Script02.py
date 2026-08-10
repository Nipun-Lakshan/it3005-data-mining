# Dataset Name :- Wine Dataset for Clustering
# Kaggle Link  :- https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering?resource=download
# Assignment   :- Analysis of Wine Dataset Using K-Means and DBSCAN Clustering Algorithms

# Import Libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load Dataset
df = pd.read_csv("wine-clustering.csv")

# Define the DataFrame as X
X = df

# Create an Object
scaler = StandardScaler()

# Learn Data and Fit The Scale
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 01. K - Means Clustering
Inertia_Values = []
SC_Values = []

# Calculate Silhouette Score & Inertia Values
for index in range(1, 9):
    data = X_pca
    Kmeans = KMeans(n_clusters=index, random_state=42)
    Kmeans.fit(data)
    Inertia_Values.append(Kmeans.inertia_)
    if index >= 2:
        sc = silhouette_score(data, Kmeans.labels_, metric='euclidean')
        SC_Values.append(sc)

# Header
print("\n========================")
print("01. K - Means Clustering")
print("========================")

# Plot Elbow Plot
plt.figure()
plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
title = 'Finding Best K Value For K Means'
plt.xlabel('K Values')
plt.ylabel('Inertia Values')
plt.title(title)
plt.grid(True)
plt.show()
BEST_K_VALUE = SC_Values.index(max(SC_Values)) + 2
print("\n01. Best K Value [Clusters]  : ", BEST_K_VALUE)
print("02. Silhouette Score (K = 3) : ", round(max(SC_Values), 2))
BEST_S_SCORE_KMEANS = round(max(SC_Values), 2)

# Plot K Means Clustering Graph
plt.figure()
Kmeans = KMeans(n_clusters=BEST_K_VALUE, random_state=42).fit(data)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Kmeans.labels_, cmap='viridis')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], marker='x', color='red')
plt.xlabel("PCA 1 Values")
plt.ylabel("PCA 2 Values")
plt.title("KMeans Plot For Wine Clustering Dataset")
plt.grid(True, alpha=0.3)
plt.show()

# Header
print("\n=====================")
print("02. DBSCAN Clustering")
print("=====================\n")

# 02. DBSCAN Clustering [Used Grid Search for Find Best Eps and Min Samples]
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