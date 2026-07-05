# Name                : A. W. W. A. Nipun Lakshan
# Registration Number : 2023s20371
# Index Number        : s17618
# Assignment          : Pairwise Feature Analysis of the Iris Dataset Using K-Means Clustering

# Import Libraries
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Clear Console
if os.name == "nt":
    subprocess.run(["cmd", "/c", "cls"])
else:
    subprocess.run(["clear"])

# Close all Previous Charts
plt.close('all')

# 01. Load Dataset & Take Features as Arrays
print("=======================")
print("Introduction to Dataset")
print("=======================\n")
df = pd.read_csv('iris.csv').drop('Id', axis=1)
print(df.head())
sepal_length = list(df['SepalLengthCm'])
sepal_width = list(df['SepalWidthCm'])
petal_length = list(df['PetalLengthCm'])
petal_width = list(df['PetalWidthCm'])
list_of_features = [sepal_length, sepal_width, petal_length, petal_width]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# 02. Draw Scatter Plots (06 Combinations)
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        plt.figure()
        plt.xlabel(feature_names[index1])
        plt.ylabel(feature_names[index2])
        title = feature_names[index1] + ' vs ' + feature_names[index2] + ' - Raw Data'
        plt.title(title)
        plt.scatter(list_of_features[index1], list_of_features[index2], s=5)
        plt.show()

# 03. Find Best KMeans Value For Raw Data
print("\n=====================")
print("Analysis for Raw Data")
print("=====================\n")
Best_K_Values = []
i = 1
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        Inertia_Values = []
        SC_Values = []
        temp_string = '0' + str(i) + '. ' + feature_names[index1] + ' vs ' + feature_names[index2]
        data = list(zip(list_of_features[index1], list_of_features[index2]))
        for index3 in range(1, 9):
            kmeans = KMeans(n_clusters=index3, random_state=0)
            kmeans.fit(data)
            Inertia_Values.append(kmeans.inertia_)
            if index3 >= 2:
                sc = silhouette_score(data, kmeans.labels_, metric='euclidean')
                SC_Values.append(sc)
        plt.figure()
        plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
        title = 'Finding Best K Value For ' + feature_names[index1] + ' vs ' + feature_names[index2] + ' - Raw Data'
        plt.xlabel('K Values')
        plt.ylabel('Inertia Values')
        plt.title(title)
        plt.grid(True)
        print(temp_string)
        print('Best K Value   : ', (SC_Values.index(max(SC_Values)) + 2))
        print('SC Values List : ', SC_Values)
        Best_K_Values.append(SC_Values.index(max(SC_Values)) + 2)
        print("\n")
        plt.show()
        i += 1
print('Raw - Best K Values : ', Best_K_Values)

# 04. Drawing Plots For KMeans
i = 0
for index5 in range(0, 4):
    for index6 in range((index5 + 1), 4):
        data = list(zip(list_of_features[index5], list_of_features[index6]))
        plt.figure()
        kmeans = KMeans(n_clusters=Best_K_Values[i], random_state=0).fit(data)
        plt.scatter(list_of_features[index5], list_of_features[index6], c=kmeans.labels_, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
        xlabel = feature_names[index5]
        ylabel = feature_names[index6]
        title = feature_names[index5] + ' vs ' + feature_names[index6] + ' - KMeans Plot - Raw Data'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
        i +=1

# 05. Data Structuring - Max
sepal_length_max = [x / max(sepal_length) for x in sepal_length]
sepal_width_max = [x / max(sepal_width) for x in sepal_width]
petal_length_max = [x / max(petal_length) for x in petal_length]
petal_width_max = [x / max(petal_width) for x in petal_width]
list_of_features = [sepal_length_max, sepal_width_max, petal_length_max, petal_width_max]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# 06. Draw Scatter Plots (06 Combinations)
print("\n=====================")
print("Analysis for Max Data")
print("=====================\n")
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        plt.figure()
        plt.xlabel(feature_names[index1])
        plt.ylabel(feature_names[index2])
        title = feature_names[index1] + ' vs ' + feature_names[index2] + ' - Structured Data - Max'
        plt.title(title)
        plt.scatter(list_of_features[index1], list_of_features[index2], s=5)
        plt.show()

# 07. Find Best KMeans Value For Max Data
Best_K_Values = []
i = 1
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        Inertia_Values = []
        SC_Values = []
        temp_string = '0' + str(i) + '. ' + feature_names[index1] + ' vs ' + feature_names[index2]
        data = list(zip(list_of_features[index1], list_of_features[index2]))
        for index3 in range(1, 9):
            kmeans = KMeans(n_clusters=index3, random_state=0)
            kmeans.fit(data)
            Inertia_Values.append(kmeans.inertia_)
            if index3 >= 2:
                sc = silhouette_score(data, kmeans.labels_, metric='euclidean')
                SC_Values.append(sc)
        plt.figure()
        plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
        title = 'Finding Best KMeans For ' + feature_names[index1] + ' vs ' + feature_names[index2] + ' - Max Data'
        plt.xlabel('K Values')
        plt.ylabel('Inertia Values')
        plt.title(title)
        plt.grid(True)
        print(temp_string)
        print('Best K Value   : ', (SC_Values.index(max(SC_Values)) + 2))
        print('SC Values List : ', SC_Values)
        print("\n")
        Best_K_Values.append(SC_Values.index(max(SC_Values)) + 2)
        plt.show()
        i += 1
print('Max - Best K Values : ', Best_K_Values)

# 08. Drawing Plots For KMeans
i = 0
for index5 in range(0, 4):
    for index6 in range((index5 + 1), 4):
        pair_data = list(zip(list_of_features[index5], list_of_features[index6]))
        plt.figure()
        kmeans = KMeans(n_clusters=Best_K_Values[i], random_state=0).fit(pair_data)
        plt.scatter(list_of_features[index5], list_of_features[index6], c=kmeans.labels_, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
        xlabel = feature_names[index5]
        ylabel = feature_names[index6]
        title = feature_names[index5] + ' vs ' + feature_names[index6] + ' - KMeans Plot - Max'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
        i += 1

# 09. Data Structuring - Min
sepal_length_min = [x / min(sepal_length) for x in sepal_length]
sepal_width_min = [x / min(sepal_width) for x in sepal_width]
petal_length_min = [x / min(petal_length) for x in petal_length]
petal_width_min = [x / min(petal_width) for x in petal_width]
list_of_features = [sepal_length_min, sepal_width_min, petal_length_min, petal_width_min]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# 10. Draw Scatter Plots (06 Combinations)
print("\n=====================")
print("Analysis for Min Data")
print("=====================\n")
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        plt.figure()
        plt.xlabel(feature_names[index1])
        plt.ylabel(feature_names[index2])
        title = feature_names[index1] + ' vs ' + feature_names[index2] + ' - Structured Data - Min'
        plt.title(title)
        plt.scatter(list_of_features[index1], list_of_features[index2], s=5)
        plt.show()

# 11. Find Best KMeans Value For Min Data
Best_K_Values = []
i = 1
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        Inertia_Values = []
        SC_Values = []
        temp_string = '0' + str(i) + '. ' + feature_names[index1] + ' vs ' + feature_names[index2]
        data = list(zip(list_of_features[index1], list_of_features[index2]))
        for index3 in range(1, 9):
            kmeans = KMeans(n_clusters=index3, random_state=0)
            kmeans.fit(data)
            Inertia_Values.append(kmeans.inertia_)
            if index3 >= 2:
                sc = silhouette_score(data, kmeans.labels_, metric='euclidean')
                SC_Values.append(sc)
        plt.figure()
        plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
        title = 'Finding Best KMeans For ' + feature_names[index1] + ' vs ' + feature_names[index2] + ' - Min Data'
        plt.xlabel('K Values')
        plt.ylabel('Inertia Values')
        plt.title(title)
        plt.grid(True)
        print(temp_string)
        print('Best K Value   : ', (SC_Values.index(max(SC_Values)) + 2))
        Best_K_Values.append(SC_Values.index(max(SC_Values)) + 2)
        print('SC Values List : ', SC_Values)
        print("\n")
        plt.show()
        i += 1
print('Min - Best K Values : ', Best_K_Values)

# 12. Drawing Plots For KMeans
i = 0
for index5 in range(0, 4):
    for index6 in range((index5 + 1), 4):
        data = list(zip(list_of_features[index5], list_of_features[index6]))
        plt.figure()
        kmeans = KMeans(n_clusters=Best_K_Values[i], random_state=0).fit(data)
        plt.scatter(list_of_features[index5], list_of_features[index6], c=kmeans.labels_, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
        xlabel = feature_names[index5]
        ylabel = feature_names[index6]
        title = feature_names[index5] + ' vs ' + feature_names[index6] + ' - KMeans Plot - Min'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
        i += 1

# 13. Data Structuring - Mean
sepal_length_mean = [x / np.mean(sepal_length) for x in sepal_length]
sepal_width_mean = [x / np.mean(sepal_width) for x in sepal_width]
petal_length_mean = [x / np.mean(petal_length) for x in petal_length]
petal_width_mean = [x / np.mean(petal_width) for x in petal_width]
list_of_features = [sepal_length_mean, sepal_width_mean, petal_length_mean, petal_width_mean]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# 14. Draw Scatter Plots (06 Combinations)
print("\n======================")
print("Analysis for Mean Data")
print("======================\n")
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        plt.figure()
        plt.xlabel(feature_names[index1])
        plt.ylabel(feature_names[index2])
        title = feature_names[index1] + ' vs ' + feature_names[index2] + ' - Structured Data - Mean'
        plt.title(title)
        plt.scatter(list_of_features[index1], list_of_features[index2], s=5)
        plt.show()

# 15. Find Best KMeans Value For Mean Data
i = 1
Best_K_Values = []
for index1 in range(0, 4):
    for index2 in range((index1+1), 4):
        Inertia_Values = []
        SC_Values = []
        temp_string = '0' + str(i)+ '. ' + feature_names[index1] + ' vs ' + feature_names[index2]
        data = list(zip(list_of_features[index1], list_of_features[index2]))
        for index3 in range(1, 9):
            kmeans = KMeans(n_clusters=index3, random_state=0)
            kmeans.fit(data)
            Inertia_Values.append(kmeans.inertia_)
            if index3 >= 2:
                sc = silhouette_score(data, kmeans.labels_, metric='euclidean')
                SC_Values.append(sc)
        plt.figure()
        plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values, color='red')
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8], Inertia_Values)
        title = 'Finding Best KMeans For ' + feature_names[index1] + ' vs ' + feature_names[index2] + ' - Mean Data'
        plt.xlabel('K Values')
        plt.ylabel('Inertia Values')
        plt.title(title)
        plt.grid(True)
        print(temp_string)
        print('Best K Value   : ', (SC_Values.index(max(SC_Values)) + 2))
        Best_K_Values.append(SC_Values.index(max(SC_Values)) + 2)
        print('SC Values List : ', SC_Values)
        print("\n")
        plt.show()
        i += 1
print('Mean - Best K Values : ', Best_K_Values)

# 16. Drawing Plots For KMeans
i = 0
for index5 in range(0, 4):
    for index6 in range((index5 + 1), 4):
        data = list(zip(list_of_features[index5], list_of_features[index6]))
        plt.figure()
        kmeans = KMeans(n_clusters=Best_K_Values[i], random_state=0).fit(data)
        plt.scatter(list_of_features[index5], list_of_features[index6], c=kmeans.labels_, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
        xlabel = feature_names[index5]
        ylabel = feature_names[index6]
        title = feature_names[index5] + ' vs ' + feature_names[index6] + ' - KMeans Plot - Mean'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
        i += 1

# 17. Silhouette Graph for Each Combination
S_Values = [0.7223033448007965, 0.7206433141776811, 0.7124025790294776, 0.43132513533997574]
X_Values = ['SL vs PW - Min', 'SW vs PW - Min', 'PL vs PW - Min', 'SL vs SW - Mean']
plt.figure()
plt.xlabel('Combination')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.5)
plt.title('Silhouette Score vs Combination')
plt.scatter(X_Values, S_Values, c='red')
plt.plot(X_Values, S_Values)
plt.show()