import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = np.array([[40, 20],[55, 30],[70, 60],[50, 35],[45, 40],[62, 75],[45, 30],[68, 80],[80, 70],[75, 90]])

# PCA Apply
X_train, X_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
print("Training Data: ", X_train)
print("Testing Data: ", X_test)

scaler = StandardScaler()
scaled_data_x_train = scaler.fit_transform(X_train)
print("Scaled Data: ", scaled_data_x_train)
scaled_data_x_test = scaler.transform(X_test)

pca = PCA(n_components=1)
pca_transformed_data_xtrain = pca.fit_transform(scaled_data_x_train)
pca_transformed_data_xtest = pca.transform(scaled_data_x_test)
print("PCA transformed Data: ", pca_transformed_data_xtrain)
print("PCA transformed Data: ", pca_transformed_data_xtest)
print("PCA Components: ", pca.components_)
print("variance: ", pca.explained_variance_ratio_)