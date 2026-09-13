# Note 07: Logistic Regression and Sigmoid Function
# =================================================

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.miscmodels.ordinal_model import OrderedModel

# 01. Plot Sigmoid Function
# =========================

x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))
plt.figure()
plt.plot(x, y)

# Plot X = 0 Type Line
plt.axvline(x = 0, color = 'red', linestyle = '--')

# Plot Y = 0 Type Line
plt.axhline(y = 0.5, color = 'green', linestyle = '--')

plt.grid(True, alpha=0.3)
plt.title("Sigmoid Function")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()

# 02. Logistic Regression - Binary Logistic Regression
# ====================================================

# Header
print("\n=============================")
print("01. Logistic Regression - BLR")
print("=============================\n")

# Load Dataset
df = pd.read_csv("Sugar_Data.csv")

# Define X and Y Varibles
x = df['Age']
y = df['AssignValue']

# Convert to NP array and Reshape
X = np.array(x).reshape(-1,1)
# -1 => Automatically Get the Count of Rows (Samples)
# 1 => One Column (One Feature)
Y = np.array(y)
# Length of X Array
print("Length of the X Array :", len(X))

# Train - Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

# Print Data
print("\nX Original Array :", df['Age'].tolist())
print("Y Original Array :", Y.tolist())
print("\nX Train          :", (x_train.reshape(1, -1).flatten()).tolist()) # flatten: Copy of 2D Array as a List
print("Y Train          :", (y_train.reshape(1, -1).flatten()).tolist()) # flatten: Copy of 2D Array as a List
print("\nX Test          :", x_test.reshape(1, -1).flatten().tolist()) # flatten: Copy of 2D Array as a List
print("Y Test          :", y_test.reshape(1, -1).flatten().tolist())  # flatten: Copy of 2D Array as a List

# Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
predicted  = model.predict(x_test)

# Print Results
print("\nThe predicted results are =", [int(x) for x in predicted])
print("The exact values for test =", [int(x) for x in y_test])

# Calculate Accuracy Score
accuracy = accuracy_score(y_test, predicted)
print("\nThe accuracy score is     =", round((accuracy * 100), 2), "\b%")
print("The accuracy score is     =", round((model.score(x_test, y_test) * 100), 2), "\b%")

# Manual Calculation
acc = (np.sum(predicted == y_test) / len(y_test)) * 100
print("\nThe accuracy score is [Manual]    = ", acc)

# Classification Report
print("\n======================")
print("Classification Report")
print("======================\n")

# Print Classification Report
print(classification_report(y_test, predicted))

# Predict a New Result
new_age = [[18]]
predction = model.predict(new_age)
probability = model.predict_proba(new_age)

print("Predcition of Sugar For Age 18  :", predction[0])
print("Probability of Sugar For Age 18 :",round((probability[0, 0] * 100), 2), "\b%")
# Proba = [[P(Class 0), P(Class 1)]]

# 03. Train - Test Split
# ======================

# Header
print("\n======================")
print("02. Train - Test Split")
print("======================\n")

# Load Dataset
df = pd.read_csv("trail.csv")
print("Column Names: ", df.columns.tolist())

# Define X and Y Variables
x = df.age
y = df.AssignValue

# Print the Type of X and Y
print("\nType of X :", type(x).__name__)
print("Type of Y :", type(y).__name__)

# Convert to Numpy Arrays
X = np.array(x)
Y = np.array(y)

# Print Types
print("\nType of X :", type(X).__name__)
print("Type of Y :", type(Y).__name__)

# NP Array Shape
a = np.array([1, 2, 3])
print("\nType of a  :", type(a).__name__)
print("Shape of a :", df.shape)

# Train - Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle = False, test_size=0.8, random_state=42)
# Shuffle: True [Best One]
print("\nLength of X Array :", len(X))
print("Length of X Train :", len(x_train))
print("Length of X Test  :", len(x_test))
print("Original X Array  :", X.tolist())
print("X Train Array     :", x_train.tolist())

# 04. Logistic Regression - Multinomial Regression
# ================================================

# Header
print("\n=============================")
print("03. Logistic Regression - MLR")
print("=============================\n")

# Dataset
data = {
    "age": [18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "income": [20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "transport": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
}

# Data Frame
df = pd.DataFrame(data)

# Features
X = df[["age", "income"]]

# Target
y = df["transport"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5,random_state=42)

# Multinomial Logistic Regression
# [multi_class="multinomial": Depreciated / Detect Automatically]
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# Predict a new person
new_person = pd.DataFrame([[32, 42]], columns = ["age", "income"])
prediction = model.predict(new_person)
probability = model.predict_proba(new_person)
print("Predicted Class:", prediction[0])
print("Probabilities:", probability[0, 1])

# 05. Logistic Regression - Ordinary Logistic Regression
# ======================================================

# Header
print("\n=============================")
print("03. Logistic Regression - OLR")
print("=============================\n")

# Dataset
data = {
    "hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    "score": [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90],
    "level": ["Low", "Low", "Low", "Medium", "Medium", "Medium", "Medium", "High", "High", "High", "High", "High"]
}

# Data Frame
df = pd.DataFrame(data)

# Convert target into ordered categories
df["level"] = pd.Categorical(
    df["level"],
    categories=["Low", "Medium", "High"],
    ordered=True)

# Features
X = df[["hours", "score"]]

# Target
y = df["level"]

# Create ordinal logistic regression model
model = OrderedModel(y, X, distr="logit")
# distr="probit" for normal distribution

# Train model
result = model.fit(method="bfgs")
# Finding the coefficients that maximize the likelihood

# Display results
print(result.summary())

# Predict
new_data = pd.DataFrame({
    "hours": [5],
    "score": [68]
})

prediction = result.model.predict(
    result.params,
    exog=new_data # Predict using new_data as the input features
)

print("\nPrediction: ", prediction)
