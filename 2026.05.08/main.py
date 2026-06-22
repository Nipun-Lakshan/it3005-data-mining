import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from packaging import markers

''' Note 01 '''
a = ([1, 2, 3, 4, 5])
print(type(a))  # List Class

''' Regression Note '''
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])
'''
plt.figure()
print(coefficients)
plt.scatter(x, y, s=20, c='red', marker='o', label='Data Points')
print(np.polyfit(x, y, 1))  # m = 2.01, c = 0.05
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
plt.grid(True)
plt.show()
'''
coefficients = np.polyfit(x, y, 1)
f = np.poly1d(coefficients)
print(f)  # Write best fitted line as a function
print(f(2.5))
yp = f(y)
print(yp)


plt.figure()
plt.scatter(x, y, s=20, c='r', marker='o', label='Original Points')
plt.scatter(x, yp, s=20, c=20, marker='x', label='Predicted Points')
plt.plot(x, yp)
plt.show()
