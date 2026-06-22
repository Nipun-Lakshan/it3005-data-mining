import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Original data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.8, 3, 6.2, 8.1, 10.3])

orders = [1, 2, 3, 4, 5]
RMSE = []

for order in orders:

    coeffe = np.polyfit(x, y, order)
    f = np.poly1d(coeffe)
    yp = f(x)
    yres = yp -y
    rmse = np.sqrt(np.mean((yres) ** 2))
