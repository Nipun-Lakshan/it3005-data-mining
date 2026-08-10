from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

df = pd.read_csv("data_set_1.csv")
X = df.drop("Category", axis=1)
Y = df["Category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3,random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)
pred = dt.predict(X_test)

print("\n01. Decision Tree Accuracy    : ", round((accuracy_score(Y_test, pred) * 100), 2))
print("02. Cross Validation Accuracy : ", round((cross_val_score(dt, X, Y, cv=5).mean() * 100), 2))

print("\nConfusion Matrix")
print("==================\n")
print(confusion_matrix(Y_test, pred))

print("\nClassification Report")
print("=====================\n")

print(classification_report(Y_test, pred))