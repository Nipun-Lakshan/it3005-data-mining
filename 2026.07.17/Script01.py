# Dataset Name :- Breast Cancer Wisconsin (Diagnostic) Data Set
# Kaggle Link  :- https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# Assignment   :- Analysis of the Breast Cancer Wisconsin Dataset Using Decision Tree and Random Forest Algorithm

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

# Load Dataset
df = pd.read_csv("BreastCancer.csv")

# Drop Unwanted Column
df = df.drop(columns=["id", "Unnamed: 32"])

# Define X and Y Variables
X = df.drop("diagnosis", axis=1)
Y = df["diagnosis"]

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Decision Tree Algorithm
decision_tree = DecisionTreeClassifier(random_state=42)

# Cross - Validation Accuracy Score
scores = cross_val_score(decision_tree, X, Y, cv=5)

print(scores)

print("Average Accuracy =", scores.mean())
