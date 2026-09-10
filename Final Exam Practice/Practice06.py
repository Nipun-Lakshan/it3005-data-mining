# Note 06: Regression Models (LR / PL / MLR)
# ==========================================

# Import Libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# 01. Linear Regression (Order 01)
# ================================

# Load Dataset
df = pd.read_csv("test_data.csv")

# Define X and Y Variables
X = df[['temperature']]
y = df['volt']

# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate Values
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print Values
print("\nOrder 01")
print("RMSE:", rmse)
print("R²  :", r2)

# 02. Polynomial Regression Model (Order > 1)
# ===========================================

# Order
degree = 3

# Mutiple Models into One Model
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate Values
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print Values
print("\nOrder 03")
print("RMSE:", rmse)
print("R²:", r2)

# 03. Cross - Validation to Evaluate Best Model
# =============================================

# Test polynomial orders
orders = [1, 2, 3, 4, 5]

# Calculate RMSE Values
for order in orders:
    model = make_pipeline(PolynomialFeatures(degree=order), LinearRegression())
    # 5-fold cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    # Convert negative RMSE to positive RMSE
    rmse = -scores
    # Print Results
    print("\nOrder:", order)
    print("RMSE for each fold:", rmse)
    print("Average RMSE:", rmse.mean())
    print()

# 04. Multiple Linear Regression
# ==============================

# Load Dataset
df = pd.read_csv("test_data.csv")

# Define X and y
X = df['temperature'].to_numpy().reshape(-1, 1)
# Feed Here More X Columns and It will Act as a MLR Problem
y = df['volt']
# -1 => Calculate Row Count Automatically / 1 => One Column

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print Results
print("RMSE:", rmse)
print("R²  :", r2)

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_[0])