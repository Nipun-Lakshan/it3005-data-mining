import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = np.array([[40, 20],[55, 30],[70, 60],[50, 35],[45, 40],[62, 75],[45, 30],[68, 80],[80, 70],[75, 90]])
plt.figure()
x = df[:,0]
y = df[:,1]
plt.scatter(x,y)
plt.show()

print("X Values:", x)
print("Y Values:", y)
scaler = StandardScaler()
scaled_data =scaler.fit_transform(df)
print("Scaled Data : ", scaled_data)

pca = PCA(n_components=2)
pca_transformed_data = pca.fit_transform(scaled_data)
print("PCA transformed Data : ", pca_transformed_data)
print("PCA Components : ", pca.components_)
print("Variance : ", pca.explained_variance_ratio_)

plt.figure()
plt.scatter(x, y, c="red")
plt.show()
plt.figure()
plt.scatter(pca_transformed_data[:,0],pca_transformed_data[:,1], c="blue")
plt.show()

# PCA Apply
X_train, X_test = train_test_split(
    df,
    test_size=0.2,
    random_state=2
)

print("Training Data: ", X_train)
print("Testing Data: ", X_test)

pca = PCA(n_components=1)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Training Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

print("\nTraining Data after PCA:")
print(X_train_pca)

print("\nTesting Data after PCA:")
print(X_test_pca)

# PCA Apply

X = np.array([
    [2.1, 1.9, 2.2, 2.0, 1.8],
    [2.3, 2.1, 2.4, 2.2, 2.0],
    [1.8, 2.0, 2.1, 1.9, 2.2],
    [2.0, 2.3, 2.5, 2.1, 2.1],
    [2.4, 2.2, 2.3, 2.4, 2.2],
    [1.9, 1.8, 2.0, 2.1, 1.9],
    [2.2, 2.4, 2.6, 2.3, 2.4],
    [2.1, 2.0, 2.3, 2.2, 2.1],
    [2.5, 2.3, 2.4, 2.5, 2.3],
    [1.8, 1.9, 2.1, 2.0, 1.8],

    [7.2, 7.0, 7.1, 6.9, 7.2],
    [7.5, 7.3, 7.4, 7.2, 7.5],
    [6.9, 7.1, 7.0, 7.2, 7.0],
    [7.3, 7.4, 7.5, 7.3, 7.4],
    [7.1, 7.2, 7.3, 7.1, 7.2],
    [6.8, 6.9, 7.0, 6.8, 6.9],
    [7.4, 7.5, 7.6, 7.4, 7.5],
    [7.0, 7.1, 7.2, 7.0, 7.1],
    [7.6, 7.4, 7.5, 7.6, 7.5],
    [6.9, 7.0, 7.1, 6.9, 7.0]
])

y = np.array([
    0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1
])