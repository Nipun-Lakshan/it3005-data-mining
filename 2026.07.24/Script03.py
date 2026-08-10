X = [
    [5.2, 24, 2.8, 62],
    [5.8, 26, 3.1, 67],
    [6.1, 28, 3.3, 71],
    [5.5, 25, 3.0, 65],
    [6.4, 30, 3.6, 75],
    [4.9, 22, 2.6, 59],
    [6.0, 27, 3.2, 69],
    [5.3, 23, 2.9, 63]
]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca)