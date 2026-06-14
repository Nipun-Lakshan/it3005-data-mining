# Decision Tree Algorithm

# Need to Complete

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = pd.read_csv("Iris.csv")
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
#
# X = np.array(x).reshape(-1, 1)
# Y = np.array(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# model = DecisionTreeClassifier(criterion="gini", max_depth=3)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# # plt.figure(figsize=(12, 8), dpi=150)
# # plot_tree(
# #     model,
# #     feature_names=iris.feature_names,
# #     class_names=list(iris.target_names),
# #     filled=True,
# #     rounded=True,
# # )
# # plt.title("Trained Decision Tree Structure")
# # plt.show()
