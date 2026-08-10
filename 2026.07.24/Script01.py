# Dataset Name :- Breast Cancer Wisconsin (Diagnostic) Data Set
# Kaggle Link  :- https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# Assignment   :- Analysis of the Breast Cancer Wisconsin Dataset Using Decision Tree and Random Forest Algorithm

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

# Load Dataset
df = pd.read_csv("BreastCancer.csv")

# Drop Unwanted Column
df = df.drop(columns=["id", "Unnamed: 32"])

# Define X and Y Variables
X = df.drop("diagnosis", axis=1)
Y = df["diagnosis"]

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 01. Decision Tree Algorithm
decision_tree = DecisionTreeClassifier(random_state=42)

# Header
print("\n============================")
print("01. Decision Tree Classifier")
print("============================\n")

# Cross - Validation Accuracy Score
scores = cross_val_score(decision_tree, X, Y, cv=5)
print("Average Cross Validation Score For Decision Tree : ", round((scores.mean() * 100),2), "%")
cross_val_accuracy_score_dt = round((scores.mean() * 100), 2)

# Train Model
decision_tree.fit(X_train, Y_train)

# Prediction
Y_Pred = decision_tree.predict(X_test)

# Accuracy
print("Prediction Accuracy of Decision Tree Classifier  : ", round((accuracy_score(Y_test, Y_Pred) * 100), 2) , "%")
accuracy_score_dt = round((accuracy_score(Y_test, Y_Pred) * 100), 2)

# Classification Report
print("\nClassification Report")
print("=====================\n")
print(classification_report(Y_test,Y_Pred))

# Confusion Matrix
print("\nConfusion Matrix")
print("================\n")
cm = confusion_matrix(Y_test,Y_Pred)
print(cm)

# Plot the Confusion Matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.show()

# 02. Random Forest Algorithm
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Header
print("\n============================")
print("02. Random Forest Classifier")
print("============================\n")

# Cross - Validation Accuracy Score
scores = cross_val_score(random_forest, X, Y, cv=5)
print("Average Cross Validation Score For Random Forest : ", round((scores.mean() * 100),2), "%")
cross_val_accuracy_score_rf = round((scores.mean() * 100), 2)

# Train Model
random_forest.fit(X_train, Y_train)

# Prediction
Y_Pred = random_forest.predict(X_test)

# Accuracy
print("Prediction Accuracy of Random Forest Classifier  : ", round((accuracy_score(Y_test, Y_Pred) * 100), 2) , "%")
accuracy_score_rf = round((accuracy_score(Y_test, Y_Pred) * 100), 2)

# Classification Report
print("\nClassification Report")
print("=====================\n")
print(classification_report(Y_test,Y_Pred))

# Confusion Matrix
print("\nConfusion Matrix")
print("================\n")
cm = confusion_matrix(Y_test,Y_Pred)
print(cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.show()

# Plot the Comparison
models = ['Decision Tree', 'Random Forest']
Cross_Validation_Score = [cross_val_accuracy_score_dt, cross_val_accuracy_score_rf]
Accuracy_Score = [accuracy_score_dt, accuracy_score_rf]

x = np.arange(len(models))
width = 0.35

plt.figure()
bars1 = plt.bar(x - width/2, Cross_Validation_Score, width, label='Cross Validation Score')
bars2 = plt.bar(x + width/2, Accuracy_Score, width, label='Test Accuracy Score')

# Display values on top of each bar
plt.bar_label(bars1, fmt='%.2f')
plt.bar_label(bars2, fmt='%.2f')

plt.xlabel("Models")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Decision Tree and Random Forest")
plt.xticks(x, models)
plt.legend(loc='center')
plt.grid(axis='y', alpha=0.3)
plt.show()