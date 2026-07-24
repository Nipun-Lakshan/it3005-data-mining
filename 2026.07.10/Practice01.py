# Augmented Data
# ==============

# The process of creating additional training data by making controlled changes
# to existing data, while preserving its essential meaning or information.

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

# Load Data
df = pd.read_csv("marks.csv")
X = df['X']
Y = df['Y']

# Question 01
# ===========

# Plot the Data
plt.figure()
plt.scatter(X, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of Marks Dataset")
plt.grid(True, alpha=0.3)
plt.show()

# Question 02
# ===========

Inertia_Values = []
SC_Values = []

for index in range(1, 9):
    data = list(zip(X, Y))
    Kmeans = KMeans(n_clusters=index, random_state=42)
    Kmeans.fit(data)
    Inertia_Values.append(Kmeans.inertia_)
    if index >= 2:
        sc = silhouette_score(data, Kmeans.labels_, metric='euclidean')
        SC_Values.append(sc)

plt.figure()
plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
title = 'Finding Best K Value For X vs Y'
plt.xlabel('K Values')
plt.ylabel('Inertia Values')
plt.title(title)
plt.grid(True)
plt.show()
BEST_K_VALUE = SC_Values.index(max(SC_Values)) + 2
print("\n01. Best K Value     : ", BEST_K_VALUE)
print("02. Silhouette Score : ", max(SC_Values))

# Question 03
# ===========

data = list(zip(X, Y))
plt.figure()
Kmeans = KMeans(n_clusters=5, random_state=42).fit(data)
plt.scatter(X, Y, c=Kmeans.labels_, cmap='viridis')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], marker='x', color='red')
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("KMeans Plot For Marks Dataset")
plt.grid(True, alpha=0.3)
plt.show()

# Question 05
# ===========

plt.figure()
plt.scatter([2, 3, 4, 5, 6, 7, 8], SC_Values, color='red')
plt.plot([2, 3, 4, 5, 6, 7, 8], SC_Values)
plt.xlabel('K Values')
plt.ylabel('Silhouette Scores')
plt.title("K Values vs Silhouette Score")
plt.grid(True, alpha=0.3)
plt.show()

# Question 06
# ===========

eps = [5, 10, 15, 20]
data = list(zip(X, Y))

plt.figure()

for index in range(4):
    clustering = DBSCAN(eps=eps[index], min_samples=30).fit(data)
    plt.subplot(2, 2, index + 1)
    plt.scatter(X, Y, c=clustering.labels_, cmap='viridis')
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.title(f"DBSCAN Plot (eps = {eps[index]})")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()