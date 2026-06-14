# Logistic Regression

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("Sugar_Data.csv")
x = data.Age
y = data.AssignValue

X = np.array(x).reshape(-1, 1)
Y = np.array(y)
print("\nLength of the X Array : ", len(X))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7)
print("X Original Array : ", X)
print("Y Original Array : ", Y)
print("X Train          : ", x_train)
print("Y Train          : ", y_train)
print("X Test           : ", x_test)
print("Y Test           : ", y_test)

# Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("The predicted results are = ", predicted)
print("The exact values for test = ", y_test)
accuracy = accuracy_score(y_test, predicted)
print("The accuracy score is     = ", accuracy)
print(model.score(x_test, y_test))

acc = (np.sum(predicted == y_test) / len(y_test)) * 100
print("The accuracy score is    = ", acc)