# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("trail.csv")
print(data.columns)

x = data.age
y = data.AssignValue
print(type(x))
print(type(y))

X = np.array(x)
Y = np.array(y)
print(type(X))
print(type(Y))

a = np.array([1, 2, 3])
print(type(a))
print(data.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle = False, test_size=0.8)
print("Length of X Array : ", len(X))
print("Length of X Train : ", len(x_train))
print("Length of X Test  : ", len(x_test))
print("Original X Array  : ", X)
print(x_train)

# The data points which used to train may be changed time to time, because shuffle will be True by default.
# Test Size will 25% by default unless it changed.
# Shuffle taking as false is not good, because variations in last data points may not be captured.

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle = False, test_size=0.8)
print("Length of X Array : ", len(X))
print("Length of X Train : ", len(x_train))
print("Length of X Test  : ", len(x_test))
print("Original X Array  : ", X)
print(x_train)

# Cross Validation
