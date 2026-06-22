import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from main import coefficients

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

coefficients = np.polyfit(x, y, 1)
f = np.poly1d(coefficients)
yp = f(x)

yres = yp - y
