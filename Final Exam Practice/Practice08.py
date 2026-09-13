# Note 08: KNN Algorithm
# ======================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

# 01. KNN Algorithm - Classification Problem
# ==========================================

# Load Dataset
df = pd.read_csv("knn_dataset.csv")

# Define X and Y Variables
x = df.drop("Class", axis=1) # axis = 1 [Column]
y = df["Class"]

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Standardization
# ===============

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# Normalization / Min - Max Scaling
# =================================

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# Find Best K Value [K Values vs Accuracy Score Graph]
k_values = range(1, 11)
accuracies = []

# Loop to Find Best K Value
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    # acc = (np.sum(y_pred == y_test) / len(y_test)) * 100
    # accuracies.append(acc) # Manual Method

# Plot and Pick the k with the highest accuracy
plt.figure()
plt.scatter(k_values, accuracies, marker = "o", label = "Accuracy Scores Data Points", color = "red")
plt.plot(k_values, accuracies, color = "blue", label = "Accuracy Scores Line")
plt.xlabel("K Values")
plt.ylabel("Accuracy Scores")
plt.title("K Values vs Accuracy Scores")
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()

# Print Best K Value
best_k = list(k_values)[np.argmax(accuracies)]
best_acc = max(accuracies)
print("\nBest K Value        : ", best_k)
print("Best Accuracy Score : ", best_acc)

# Create KNN Model
knn = KNeighborsClassifier(n_neighbors = 2, weights = "uniform", metric = "minkowski", algorithm = "auto", leaf_size = 30, p=2, n_jobs = None, metric_params=None)

# Fit the Model (Train the Model)
knn.fit(x_train, y_train)

# Predict
y_pred = knn.predict(x_test)

# Evaluation
print("\n==================================")
print("Evaluation Report - KNN Classifier")
print("==================================\n")

print("Accuracy Score as a Percentage:", round((accuracy_score(y_test, y_pred) * 100), 2), "\b%")

# Confusion Matrix
print("\n================")
print("Confusion Matrix")
print("================\n")
print(confusion_matrix(y_test, y_pred), "\n")

# Predict a New One
new_one = pd.DataFrame([[6.1, 5.9, 5.6]], columns = ["Feature1", "Feature2", "Feature3"])
prediction = knn.predict(new_one)

# Print Prediction Result
print("Expected Class  : Class 1")
print("Predicted Class : Class", prediction[0])

# 02. KNN Algorithm - Regression Problem
# ======================================

# Create sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10] ]) # 10 x 1 2D Matrix
y = np.array([10, 15, 20, 24, 30, 35, 40, 45, 50, 55])

# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Regressor
knn = KNeighborsRegressor(n_neighbors=3, weights='uniform')

# Train
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print Results
print("\n=================================")
print("Evaluation Report - KNN Regressor")
print("=================================\n")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)