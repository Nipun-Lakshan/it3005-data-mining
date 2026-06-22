import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

# Find coefficients of best fitted line
coefficients = np.polyfit(x, y, 1)

# Slope and intercept
m = coefficients[0]
c = coefficients[1]

print("Slope :", m)
print("Intercept :", c)

# Equation of fitted line
y_fit = m * x + c

# Plot original points
plt.scatter(x, y, label="Data Points")

# Plot best fitted line
plt.plot(x, y_fit, label="Best Fitted Line")

# Labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best Fitted Line")

plt.legend()
plt.grid(True)

plt.show()