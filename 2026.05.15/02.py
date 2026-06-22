# Regression Method 02 : R Squared Method

import numpy as np
import matplotlib.pyplot as plt

# Original data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

orders = range(1, 6)
R2 = []

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

    r_squared = 1 - (ss_res / ss_tot)

    R2.append(r_squared)

# Plot R² vs Order
plt.figure()

plt.plot(orders, R2, label="R²")
plt.scatter(orders, R2, marker="o", color="r")

plt.xlabel("Order")
plt.ylabel("R²")
plt.title("R² vs Polynomial Order")

plt.grid(True)
plt.show()

# Plot original data
plt.figure()

plt.scatter(x, y, label="Original Data", marker="o", color="r")
plt.plot(x, y, label="Original Data Line", color="b")

# Best fitted polynomial
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