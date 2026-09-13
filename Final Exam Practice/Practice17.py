# Note 17: Data Cleaning Methods [MA, GF & MF]
# ============================================

# Import Libraries
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

# 01. 7MA Code
# ============

# Define X values
X = np.arange(1, 23)

# Define Y values
Y_AD1 = np.array([
    10, 12, 15, 14, 18, 20, 22,
    19, 21, 25, 24, 28, 30, 27,
    29, 32, 35, 33, 36, 38, 40, 42
])

# Prepare X Values for MA-7
X_MA7 = X[0:-6]

# Calculate 7-point Moving Average
Y_MD1 = np.convolve(Y_AD1, np.ones(7), 'valid') / 7

# Print Results
print("\nX Values:", X_MA7)
print("MA-7 Values:", Y_MD1)

# 02. Median Filter
# =================

# Numerical data
data = np.array([10, 12, 100, 14, 15, 13, 200, 16, 18])

# Apply Median Filter
filtered_data = medfilt(data, kernel_size=3)

# Print results
print("\nOriginal Data:")
print(data)

print("Filtered Data:")
print(filtered_data)

# 03. Gaussian Filter
# ===================

# Numerical data
data = np.array([10, 12, 15, 100, 14, 16, 18, 200, 20])

# Apply Gaussian Filter
filtered_data = gaussian_filter1d(data, sigma=1)

# Print results
print("\nOriginal Data:")
print(data)

print("Gaussian Filtered Data:")
print(filtered_data)