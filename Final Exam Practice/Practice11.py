# Note 11: Naive Bayes Algorithm
# ==============================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import BernoulliNB

# Converts text into numerical feature vectors by counting word occurrences
# Mainly used to convert text data into numbers for NLP / ML models.
from sklearn.feature_extraction.text import CountVectorizer

# ========================
# 01. Gaussian Naive Bayes
# ========================

# Load dataset
df = pd.read_csv("iris.csv")

# Separate features and target
X = df.drop(columns=["Species", "Id"])
y = df["Species"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print Header
print("\n========================")
print("01. Gaussian Naive Bayes")
print("========================\n")
print("01. Accuracy : ", round((accuracy * 100), 2))

# Confusion Matrix
print("\n================")
print("Confusion Matrix")
print("================\n")

# Print Result
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\n=====================")
print("Classification Report")
print("=====================\n")
print(classification_report(y_test, y_pred))

# Predict a new sample
new_data = pd.DataFrame([[25, 50000, 170, 1000]],columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
prediction = model.predict(new_data)
print("\nPrediction:", prediction)

# ===========================
# 02. Multinomial Naive Bayes
# ===========================

# Load dataset
# df = pd.read_csv("data.csv")

# Example columns:
# text   -> message/email
# target -> spam/not spam

# Separate input and target
# X = df["text"]
# y = df["target"]

# Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text into word-count features
# vectorizer = CountVectorizer()
# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)

# Create Multinomial Naive Bayes model
# model = MultinomialNB()

# Train model
# model.fit(X_train, y_train)

# Make predictions
# y_pred = model.predict(X_test)

# Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Confusion Matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# Predict new text
# new_text = ["Congratulations! You won a free prize"]
# new_text = vectorizer.transform(new_text)
# prediction = model.predict(new_text)
# print("\nPrediction:", prediction)

# =========================
# 03. Bernoulli Naive Bayes
# =========================

# Load dataset
# df = pd.read_csv("data.csv")

# Example:
# free, offer, win -> binary features
# target -> class

# Separate features and target
# X = df.drop("target", axis=1)
# y = df["target"]

# Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Bernoulli Naive Bayes model
# model = BernoulliNB()

# Train model
# model.fit(X_train, y_train)

# Make predictions
# y_pred = model.predict(X_test)

# Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Confusion Matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# Predict a new sample
# new_data = [[1, 1, 0]]
# prediction = model.predict(new_data)
# print("\nPrediction:", prediction)