# Group Assignment 01 - 2026.06.05

# Classification Model : Random Forest
# Dataset : Iris.csv

# Team Members
# ============

# 01. 2023s20371 - s17618 - A. W. W. A. Nipun Lakshan
# 02. 2023s20376 - s17601 - T. W. Himal Rasanjana
# 03. 2023s19911 - s17397 - K. R. D. Fernando

# 01. Import libraries

# Import the pandas library and give it the alias 'pd'.
# Pandas is used for reading and manipulating datasets in tabular form. (DataFrames)
import pandas as pd

# Import the train_test_split function.
# Used to divide the dataset into training data and testing data.
from sklearn.model_selection import train_test_split

# Import the RandomForestClassifier class.
# Used to create and train a Random Forest machine learning model for classification tasks.
from sklearn.ensemble import RandomForestClassifier

# Import evaluation metrics
# accuracy_score        -> Calculates the overall prediction accuracy
# classification_report -> Displays precision, recall, f1-score and support for each class
# confusion_matrix      -> Shows correct and incorrect predictions in matrix form
from sklearn.metrics import accuracy_score, confusion_matrix

# 02. Load the Iris dataset from the CSV file into a DataFrame
iris = pd.read_csv("Iris.csv")

# 03. Select the feature columns by removing the target column (Species) and the ID column which is not useful for prediction
X = iris.drop(columns=["Species", "Id"])

# 04. Store the target variable (flower species) in y
y = iris["Species"]

# 05. Split the dataset into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size    => Specifies the size of the test set. (As a percentage)
# train_size   => Specifies the size of the training set.
# random_state => Controls the random shuffling by setting a random seed
# shuffle      => Whether to shuffle data before splitting. (Default => True)
# stratify     => Maintains the class distribution in both train and test sets.

# 06. Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# n_estimators => Number of decision trees in the forest. (Larger value → usually better accuracy but slower training)
# criterion
# =========
# 'gini'     - Measures how often a randomly chosen element would be misclassified if it was labeled randomly according to class distribution.
# 0 → pure node (perfect) / Higher value → more mixed classes - Default
# 'entropy'  - Measures uncertainty or disorder in the data. [0 → no uncertainty (Pure Node) / Higher value → more mixed classes]
# 'log_loss' - Measures how wrong probability predictions are.
# max_depth -> Maximum depth of each tree. [None -> Grow Endlessly / Smaller values reduce overfitting]
# min_samples_split -> Minimum samples required to split a node. [Normaly 10]
# min_samples_leaf -> Minimum samples required in a leaf node. [Minimum 5]
# max_features -> Number of features considered when finding the best split.
# bootstrap -> Whether sampling is done with replacement. [Default True]
# random_state -> Controls randomness. [Default - None / Value means same result every run]
# class_weight -> Weights classes differently. [Default -> None / Weight -> Balanced]

# 07. Train model
model.fit(X_train, y_train)

# 08. Predict
y_pred = model.predict(X_test)

# 09. Evaluation
print("\n===========================================")
print("Evalution Report - Random Forest Classifier")
print("===========================================\n")
print(f"Accuracy as a Percentage : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("================")
print("Confusion Matrix")
print("================\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

# 10. Predict a New Flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
prediction = model.predict(new_flower)
print("Expected Flower  :  Iris-setosa")
print("Predicted Flower : ", prediction[0])

# Inside Story
# ============

# 01. Many decision trees are created.
# Example: 100 trees
# 02. Each tree gets different data.
# Random samples of rows (bootstrap sampling)
# 03. At each split, only some features are used
# You have 4 features
# Each split randomly selects 2 features (√4)
# Each tree learns its own rules
# Different trees look different
# 04. Prediction happens by voting
# Each tree predicts a class.
# Final answer = majority vote