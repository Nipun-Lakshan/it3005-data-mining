# Note 02: Linear Regression (RMSE Method)
# ========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 01. Load Dataset
# ================

df = pd.read_csv('test_data.csv')

# 02. Define X and Y Varibales
# ============================

X = df['temperature'].tolist()
Y = df['volt'].tolist()
print("\n01. Type of X : ", type(X).__name__)
print("02. Type of Y : ", type(Y).__name__)

# 03. Draw a Scatter Plot to Identify the Relationship
# ====================================================

plt.figure()
plt.scatter(X, Y, marker='o', label='Original Data', c='red', s=20)
plt.plot(X, Y, label='Original Data', c='blue')
plt.title('Temperature vs Voltage (Original Data)')
plt.xlabel('Temperature')
plt.ylabel('Voltage')
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()

# Print Coefficients of Order 1
coefficients = np.polyfit(X, Y, 1)
print("03. Slope of the Polynomial     [Order 01] : ", coefficients[0])
print("04. Intercept of the Polynomial [Order 01] : ", coefficients[1])

# 04. Draw Elbow Plot to Find Best RMSE Vlaue
# ===========================================

# Function to Draw Elbow Plot
def draw_elbow_plot(start, end, rmse_values, orders, x, y):

    for order in range(start, end): # start <= order < end
        # Fit polynomial
        coefficients = np.polyfit(x, y, order)
        # Create polynomial function
        polynomial = np.poly1d(coefficients)
        # Predicted y values
        y_pred = polynomial(x)
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        # Store values
        orders.append(order)
        rmse_values.append(rmse)

    # Plot the Order vs RMSE Graph
    plt.figure()
    plt.scatter(orders, rmse_values, marker='o', s = 20, c='red')
    plt.plot(orders, rmse_values, c='blue')
    plt.xlabel('Order')
    plt.ylabel('RMSE')
    plt.title('Order vs RMSE')
    plt.xticks()
    plt.grid(True, alpha = 0.3)
    plt.plot()
    plt.show()

# Call to the Function
rmse_values = []
orders = []
draw_elbow_plot(1, 9, rmse_values, orders, X, Y)
rmse_values = [float(x) for x in rmse_values]
print("05. RMSE List  :", rmse_values)
print("06. Order List :", orders)
print("07. Best Order : 3") # From Looking at Elbow Method

# Function to Plot Smooth Curve for Best Fitted Line
def smooth_curve(x, y, best_order, intervals):
    coefficients = np.polyfit(x, y, best_order)
    f = np.poly1d(coefficients)
    x_new = np.linspace(min(x), max(x), intervals)
    y_new = f(x_new)
    plt.figure()
    plt.scatter(x, y, marker='o', label='Original Data', c='red', s=20)
    plt.plot(x_new, y_new, c='blue', label='Smooth Curve')
    plt.title('Best Fitted Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.show()

    # Find Y Value For Given X
    print("Y value when X = 33 : ", f(33))

# Call a Function to Plot Smooth Curve
smooth_curve(X, Y, 3, 10000)

# 05. Find X Value For Given Y [Inverse Function]
# ===============================================

# Define X and Y by Changing the Values
X_inv = Y
Y_inv = X

# Define Two Empty Lists to Store RMSE and Orders
rmse_values_inv = []
orders_inv = []

# Call Two Functions to Get Best Fitted Line For Inverse Function
draw_elbow_plot(1, 9, rmse_values_inv, orders_inv, X_inv, Y_inv)
smooth_curve(X_inv, Y_inv, 3, 10000)