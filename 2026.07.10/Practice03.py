import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = np.array([
    [40,20],
    [55,30],
    [70,60],
    [50,35],
    [45,40],
    [62,75],
    [45,30],
    [68,80],
    [80,70],
    [75,90]
])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_data)

print(pca.explained_variance_ratio_)
print(pca.components_)

plt.figure()
plt.scatter(df[:,0],df[:,1])
plt.title('RAW')
plt.show()
# See raw and PCA graphs
# plt.figure()
# plt.scatter(scaled_data.iloc[:,0],scaled_data.iloc[:,1])
# plt.title('RAW')
# plt.show()