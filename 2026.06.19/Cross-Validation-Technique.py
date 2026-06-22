# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 01. Dataset Preparation
iris = pd.read_csv("Iris.csv")
X = iris.drop(columns=["Species", "Id"])
y = iris["Species"]

# 02. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

# 03. Create KNN model
model1 = KNeighborsClassifier(n_neighbors=3)

# 04. Train model
model1.fit(X_train, y_train)

# 05. Predict
y_pred = model1.predict(X_test)

# 06. Accuracy Score
print(f"Accuracy as a Percentage for KNN Model : {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 07. Create Naive Bayes model
model2 = GaussianNB()

# 08. Train model
model2.fit(X_train, y_train)

# 09. Predict
y_pred = model2.predict(X_test)

# 10. Accuracy Score
print(f"Accuracy as a Percentage for Naive Bayes Model : {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 11. Cross Validation
scores = cross_val_score(model1, X, y, cv=5)
print("Cross Validation Mean For KNN Model : ", (np.mean(scores) * 100))

# 12. SVM Model
model3 = SVC(kernel="rbf", gamma="auto")
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
print(f"Accuracy as a Percentage for Naive Bayes Model : {accuracy_score(y_test, y_pred) * 100:.2f}%")
scores = cross_val_score(model3, X, y, cv=5)
print("Cross Validation Mean For SVM : ", (np.mean(scores) * 100))