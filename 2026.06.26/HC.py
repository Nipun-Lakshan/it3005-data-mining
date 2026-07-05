import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc

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
