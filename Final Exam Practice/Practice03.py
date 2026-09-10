# Note 03: Finding the Best Fitted Line (R_Squared Method / Adjusted R_Squared Value)
# ===================================================================================

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset
df = pd.read_csv("test_data.csv")

# Define X and Y Variables
x = df['temperature']
y = df['volt']

# Define Empty Lists To Store R2 Values and Define the Order
R2 = []
orders = range(1, 6)

# Draw Elbow Plot
for order in orders:
    # Fit polynomial
    coefficient = np.polyfit(x, y, order)
    # Create polynomial function
    f = np.poly1d(coefficient)
    # Predicted y values
    yp = f(x)
    # R² calculation
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # Finding R_Squared
    r_squared = 1 - (ss_res / ss_tot)
    # Append R_Squred Valued
    R2.append(r_squared)

# Plot R² vs Order
plt.figure()
plt.scatter(orders, R2, label="R²", color="r", marker="o")
plt.plot(orders, R2, label="R²", color="b")
plt.xlabel("Order")
plt.ylabel("R²")
plt.title("R² vs Polynomial Order")
plt.grid(True, alpha=0.3)
plt.show()

# Print Best Order
print("Best Order : 3")

# Plot the Smooth Curve
plt.figure()
plt.scatter(x, y, label="Original Data Points", marker="o", color="r")
x_new = np.linspace(min(x), max(x), 1000)
coefficient = np.polyfit(x, y, 3)
f = np.poly1d(coefficient)
yp = f(x_new)
plt.plot(x_new, yp, label="Best Fitted Line", color="g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best Fitted Polynomial")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Find Adjusted R_Squared Values
R_Squared_Value = R2[2]
n = 31 # Number of Observations
p = 1 # Number of Independent Variables
Adjusted_R_Squared_Value = (1 - ((1 - R_Squared_Value)*((n - 1)/(n - p - 1))))
print("Adjusted R_Squared_Value:", Adjusted_R_Squared_Value)