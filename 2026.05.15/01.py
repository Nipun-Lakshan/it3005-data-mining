# Regression Method 01 : RMSE Method

import numpy as np
import matplotlib.pyplot as plt

# Original data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

orders = range(1,6)
RMSE = []

for order in orders:

    # Fit polynomial
    coefficient = np.polyfit(x, y, order)

    # Create polynomial function
    f = np.poly1d(coefficient)

    # Predicted y values
    yp = f(x)

    # RMSE calculation
    rmse = np.sqrt(np.mean((y - yp) ** 2))

    RMSE.append(rmse)

plt.figure()
plt.plot(orders, RMSE, label="RMSE")
plt.scatter(orders, RMSE, label="RMSE", marker="o", color="r")
plt.xlabel("Order")
plt.ylabel("RMSE")
plt.title("RMSE vs Order")
plt.grid(True)
plt.show()

plt.figure()
# Plot original data
plt.scatter(x, y, label="Original Data", marker="o", color="r")
plt.plot(x, y, label="Original Data Line", color="b")

# Smooth curve for best fit
x_new = np.linspace(min(x), max(x), 1000)
coefficient = np.polyfit(x, y, 4)
f = np.poly1d(coefficient)
yp = f(x_new)
plt.plot(x_new, yp, label="Best Fitted Line", color="g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best Fitted Polynomial")
plt.grid(True)
plt.legend()
plt.show()