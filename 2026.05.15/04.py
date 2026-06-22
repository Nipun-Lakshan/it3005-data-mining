# Classification Technique - Sigmoid Function

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.grid(True)
plt.title("Sigmoid Function")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()
