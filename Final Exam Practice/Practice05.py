# Note 05: Finding the Best Fitted Line (MSE Method)
# ==================================================

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset
df = pd.read_csv("test_data.csv")

# Define X and Y Varibales
x = df['temperature']
y = df['volt']

# Plot the Elbow Graph
orders = [1, 2, 3, 4, 5, 6]
MSE = []

# Calculate MAE Values
for order in orders:
    # Fit polynomial
    coefficient = np.polyfit(x, y, order)
    # Create polynomial function
    f = np.poly1d(coefficient)
    # Predicted y values
    yp = f(x)
    # MAE calculation
    mse = np.mean((y - yp)**2)
    # Append Values
    MSE.append(mse)

# Plot the Graphs
plt.figure()
plt.plot(orders, MSE, label="MSE", color="b")
plt.scatter(orders, MSE, label="MAE", marker="o", color="r")
plt.xlabel("Order")
plt.ylabel("MSE")
plt.title("MSE vs Order")
plt.grid(True, alpha=0.3)
plt.show()

# Plot the Smooth Curve
plt.figure()
plt.scatter(x, y, label="Original Data", marker="o", color="r")
x_new = np.linspace(min(x), max(x), 1000)
coefficient = np.polyfit(x, y, 4)
f = np.poly1d(coefficient)
yp = f(x_new)
plt.plot(x_new, yp, label="Best Fitted Line", color="g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best Fitted Polynomial")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Get Minimum MSE Order
print("Best Order : " + str(np.argmin(MSE[0:3])+1))