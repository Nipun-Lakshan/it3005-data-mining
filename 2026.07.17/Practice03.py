import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# print(X.shape)
# print(y.shape)

X_train, X_test, Y_train, Y_test  = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
# Ensures the same proportion of each class is maintained in both
    # the training and testing datasets.
)

print("Training Data X          : ", X_train)
print("Training Data Shape of X : ", X_train.shape)
print("Test Data X              : ", X_test)
print("Test Data Shape of X     : ", X_test.shape)

scaler = StandardScaler()
scaled_data_x_train = scaler.fit_transform(X_train)
scaled_data_x_test = scaler.transform(X_test)

pca = PCA(n_components=3)
pca_data_x_train = pca.fit_transform(scaled_data_x_train)
pca_data_x_test = pca.transform(scaled_data_x_test)
print("PCA X Train transformed Data: ", pca_data_x_train)
print("PCA X test transformed Data: ", pca_data_x_test)
print("PCA Components: ", pca.components_)
print("variance: ", pca.explained_variance_ratio_)

# plt.figure()
# plt.scatter(pca_data[:, 0], pca_data[:, 1], c='red')
# plt.show()

#
# pca = PCA(n_components=1)
# pca_transformed_data_xtrain = pca.fit_transform(scaled_data_x_train)
# pca_transformed_data_xtest = pca.transform(scaled_data_x_test)

