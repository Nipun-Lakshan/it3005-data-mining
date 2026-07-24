import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x = np.array([
    [5.1, 3.5, 1.4, 0.2, 0.50],
    [4.9, 3.0, 1.4, 0.2, 0.45],
    [4.7, 3.2, 1.3, 0.2, 0.48],
    [4.6, 3.1, 1.5, 0.2, 0.52],
    [5.0, 3.6, 1.4, 0.3, 0.55],
    [5.4, 3.9, 1.7, 0.4, 0.60],
    [4.8, 3.4, 1.6, 0.2, 0.50],
    [5.0, 3.4, 1.5, 0.2, 0.53],
    [4.4, 2.9, 1.4, 0.2, 0.47],
    [4.9, 3.1, 1.5, 0.1, 0.49],

    [6.4, 3.2, 4.5, 1.5, 1.20],
    [6.9, 3.1, 4.9, 1.5, 1.30],
    [5.5, 2.3, 4.0, 1.3, 1.10],
    [6.5, 2.8, 4.6, 1.5, 1.25],
    [5.7, 2.8, 4.5, 1.3, 1.15],
    [6.3, 3.3, 4.7, 1.6, 1.28],
    [4.9, 2.4, 3.3, 1.0, 0.95],
    [6.6, 2.9, 4.6, 1.3, 1.22],
    [5.2, 2.7, 3.9, 1.4, 1.05],
    [5.0, 2.0, 3.5, 1.0, 0.98]
])

y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print("\nScaled X Data : ", x_scale)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scale)
print("\nNew Data Set After PCA : ", x_pca)
print("\nShape of the Dataset   : ", x_pca.shape)
print("\nVariance of the PCA    : ", (pca.explained_variance_ratio_*100))
print("\nComponents of the PCA  : ", pca.components_)

plt.figure()
plt.scatter(x_pca[y==0, 0], x_pca[y==0, 1], c='red', label='Class 0')
plt.scatter(x_pca[y==1, 0], x_pca[y==1, 1], c='green', label='Class 0')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()

svm = SVC(kernel='linear')
svm.fit(x_pca, y)

new_dataset = np.array([[6.4, 3.2, 4.5, 1.5, 1.20]])
print("\nTest Sample : ", new_dataset)
x_scale = scaler.transform(new_dataset)
print("\nNew Scaled Test Data   : ", x_scale)

new_x_pca = pca.transform(x_scale)

print("\nNew Data Set After PCA : ", x_pca)
print("\nShape of the Dataset   : ", x_pca.shape)
print("\nVariance of the PCA    : ", (pca.explained_variance_ratio_*100))
print("\nComponents of the PCA  : ", pca.components_)

print("\nClass : ", svm.predict(new_x_pca)[0])

print(np.__version__)
P = np.array([[6.4, 3.2, 4.5, 1.5, 1.20], [5.0, 3.6, 1.4, 0.2, 0.5]])
print(P[0, int(True)])
print(P[0, True])