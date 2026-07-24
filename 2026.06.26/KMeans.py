import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(X, y)
plt.grid(True)
plt.show()
data = list(zip(X, y))

print(data)

kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(data)
kmeans.fit(data)
print("Cluster Labels  : ", kmeans.labels_)
print("Cluster Centers : {", kmeans.cluster_centers_[0], ", ", kmeans.cluster_centers_[1], "}")

plt.figure()
plt.scatter(X, y, c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
plt.grid(True)
plt.show()
print(type(kmeans.cluster_centers_))

Inertia_Values = []
for key in range(1, 6):
    kmeans = KMeans(n_clusters=key, random_state=0)
    kmeans.fit(data)
    Inertia_Values.append(kmeans.inertia_)

plt.figure()
plt.scatter(range(1, 6), Inertia_Values, color='red')
plt.plot(range(1, 6), Inertia_Values)
plt.show()

sc = silhouette_score(data, kmeans.labels_, metric='euclidean')
print(sc)

sc_values = []
for key in range(2, 6):
    kmeans = KMeans(n_clusters=key, random_state=0)
    kmeans.fit(data)
    sc_values.append(silhouette_score(data, kmeans.labels_))
print("SC Values : ", sc_values)