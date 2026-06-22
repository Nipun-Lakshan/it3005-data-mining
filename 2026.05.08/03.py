import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Original data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

orders = []
rmse_values = []

# Polynomial orders from 1 to 5
for order in range(1, 9):

    # Fit polynomial
    coefficients = np.polyfit(x, y, order)

    # Create polynomial function
    polynomial = np.poly1d(coefficients)

    # Predicted y values
    y_pred = polynomial(x)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Store values
    orders.append(order)
    rmse_values.append(rmse)

# Plot
plt.plot(orders, rmse_values, marker='o')

plt.xlabel("Order of Polynomial")
plt.ylabel("RMSE Value")
plt.title("Order vs RMSE")

plt.xticks(orders)
plt.grid(True)

plt.show()