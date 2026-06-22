# KNN Algorithm

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("Sugar_Data.csv")
x = data.Age
y = data.AssignValue

X = np.array(x).reshape(-1, 1)
Y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

accuracy = [];

for variable in range(1, 6):
    model = KNeighborsClassifier(n_neighbors=variable)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = (np.sum(y_pred == y_test) / len(y_test)) * 100
    accuracy.append(acc)

x = [1, 2, 3, 4, 5]

plt.plot(x, accuracy)
plt.xlabel("K Value")
plt.ylabel("Accuracy Score")
plt.title("Finding Best K Value")
plt.grid(True)
plt.show()