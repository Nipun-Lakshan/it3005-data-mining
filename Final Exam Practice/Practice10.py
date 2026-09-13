# Note 10: Random Forest Algorithm
# ================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = pd.read_csv("Iris.csv")

# Features and Target
X = iris.drop(columns=["Species", "Id"])
y = iris["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\n===========================================")
print("Evalution Report - Random Forest Classifier")
print("===========================================\n")
print(f"Accuracy as a Percentage : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("================")
print("Confusion Matrix")
print("================\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

# Predict a New Flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
prediction = model.predict(new_flower)
print("Expected Flower  :  Iris-setosa")
print("Predicted Flower : ", prediction[0])

# Print Header
print("\n==================")
print("Important Features")
print("==================\n")

# Find Important Features
for feature, importance in zip(X_train.columns, model.feature_importances_):
    print(feature, ":", importance)