import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("data_set_1.csv")
X = df.drop("Category", axis=1)
Y = df["Category"]

dt = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(dt, X, Y, cv=5)
print("\n01. Cross Validation Accuracy Score For Decision Tree : ", round((scores.mean() * 100), 2), "\b%")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
dt.fit(X_train, Y_train)
print("02. Test Accuracy Score For Decision Tree             : ", round((accuracy_score(Y_test, (dt.predict(X_test))) * 100), 2), "\b%")

for i in [3, 2, 1]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X_scaled)
    print("\nPCA Component : ", i)

    Inertia_Values = []
    SC_Values = []

    for index in range(1, 9):
        data = X_pca
        Kmeans = KMeans(n_clusters=index, random_state=42)
        Kmeans.fit(data)
        Inertia_Values.append(Kmeans.inertia_)
        if index >= 2:
            sc = silhouette_score(data, Kmeans.labels_, metric='euclidean')
            SC_Values.append(sc)

    plt.figure()
    plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
    title = 'Finding Best K Value For K Means' + ' [PCA - ' + str(i) + ']'
    plt.xlabel('K Values')
    plt.ylabel('Inertia Values')
    plt.title(title)
    plt.grid(True)
    plt.show()
    BEST_K_VALUE = SC_Values.index(max(SC_Values)) + 2
    print("\n01. Best K Value [Clusters]   : ", BEST_K_VALUE)
    print("02. Silhouette Score (K = ", BEST_K_VALUE, "\b) : ", round(max(SC_Values), 2))
    BEST_S_SCORE_KMEANS = round(max(SC_Values), 2)

    # Plot K Means Clustering Graph
    if i == 2:
        plt.figure()
        Kmeans = KMeans(n_clusters=BEST_K_VALUE, random_state=42).fit(data)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Kmeans.labels_, cmap='viridis')
        plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], marker='x', color='red')
        plt.xlabel("PCA 1 Values")
        plt.ylabel("PCA 2 Values")
        plt.title("KMeans Plot For PCA 2")
        plt.grid(True, alpha=0.3)
        plt.show()

    # Plot K Means Clustering Graph
    if i == 3:
        # 1. Create figure and 3D axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 2. Generate sample data
        z = X_pca[:, 2]
        x = X_pca[:, 0]
        y = X_pca[:, 1]

        # 3. Plot data (e.g., 3D line or scatter)
        ax.scatter3D(x, y, z, c=Kmeans.labels_, cmap='viridis')

        # 4. Display
        plt.show()
