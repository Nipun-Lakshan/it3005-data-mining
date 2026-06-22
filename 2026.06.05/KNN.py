# Group Assignment 01 - 2026.06.05

# Classification Model : K-Nearest Neighbors (KNN)
# Dataset : Iris.csv

# Team Members
# ============

# 01. 2023s20371 - s17618 - A. W. W. A. Nipun Lakshan
# 02. 2023s20376 - s17601 - T. W. Himal Rasanjana
# 03. 2023s19911 - s17397 - K. R. D. Fernando

# 01. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 02. Load dataset
iris = pd.read_csv("Iris.csv")

# 03. Features and target
X = iris.drop(columns=["Species", "Id"])
y = iris["Species"]

# 04. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# 05. Create KNN model
model = KNeighborsClassifier(n_neighbors=3, )

# 06. Train model
model.fit(X_train, y_train)

# 07. Predict
y_pred = model.predict(X_test)

# 08. Evaluation
print("\n==================================")
print("Evaluation Report - KNN Classifier")
print("==================================\n")
print(f"Accuracy as a Percentage : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("================")
print("Confusion Matrix")
print("================\n")
print(confusion_matrix(y_test, y_pred), "\n")

# 09. Predict a New Flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

prediction = model.predict(new_flower)

print("Expected Flower  :  Iris-setosa")
print("Predicted Flower : ", prediction[0])