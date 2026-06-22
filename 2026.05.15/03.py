# Regression Method 03 : MAE Method

import numpy as np
import matplotlib.pyplot as plt

# Original data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

orders = range(1,6)
MAE = []

for order in orders:

    # Fit polynomial
    coefficient = np.polyfit(x, y, order)

    # Create polynomial function
    f = np.poly1d(coefficient)

    # Predicted y values
    yp = f(x)

    # MAE calculation
    mae = np.mean(np.abs(y - yp))

    MAE.append(mae)

plt.figure()
plt.plot(orders, MAE, label="MAE")
plt.scatter(orders, MAE, label="MAE", marker="o", color="r")
plt.xlabel("Order")
plt.ylabel("MAE")
plt.title("MAE vs Order")
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