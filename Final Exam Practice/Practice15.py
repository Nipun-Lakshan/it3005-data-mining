# Note 15: Hierarchial Clustering
# ===============================

# Import Libraries
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc

# ======================================
# 01. HC  Agglomerative Clustering Model
# ======================================

# Define Data
X = np.array([1, 2, 3, 5, 6, 8, 9]).reshape(-1, 1)

# Create a Model
model = AgglomerativeClustering(n_clusters=3, linkage="ward")

# Fit & Predict From Model
labels = model.fit_predict(X)

# Print Predictions
print(labels)

# =============
# 02. Dendogram
# =============

X = np.array([1, 2, 3, 5, 6, 8, 9])
y = np.array([2, 3, 1, 7, 8, 7, 9])

h = list(range(len(X)))

for i, v in enumerate(h):
    plt.annotate(v, xy=(X[i], y[i]))

data = list(zip(X, y))

z = hc.linkage(data, method='ward')

plt.figure()
hc.dendrogram(z)
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()