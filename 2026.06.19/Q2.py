import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 01. Load Dataset
data = pd.read_csv('pendulum.csv')
print("\n")

# 02. Calculate Mean Time Period For Each Length
Mean_Time_Period = []
for key in range(1, 11):
    Mean_Time_Period.append(float(data.iloc[0:9, key].mean()))
STD_Time_Period = []
for key in range(1, 11):
    STD_Time_Period.append(float(data.iloc[0:9, key].std()))
print(Mean_Time_Period)
print(STD_Time_Period)

# 03. Plot the Dataset
mean_time = np.square(np.array(Mean_Time_Period))
length = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.figure()
plt.scatter(length, mean_time, label="Mean_Time_Period")
plt.grid(True)
plt.xlabel("Length (cm)")
plt.ylabel("Mean Time Period (T²)")
plt.title("Mean Time Period vs Length")
plt.show()

# 04. Plot an Error Bar
plt.figure()
plt.errorbar(length, mean_time, yerr=STD_Time_Period, label="Mean with SE", capsize=5, fmt='o')
plt.grid(True)
plt.xlabel("Length (cm)")
plt.ylabel("Mean Time Period (T²)")
plt.title("Error Bar Plot")
plt.legend()
plt.show()

# 05. Fit a plot without errors
plt.figure()
coefficients = np.polyfit(length, mean_time, 1)

# Fit Without Errors
f = np.poly1d(coefficients)
X = np.linspace(min(length), max(length), 1000)
plt.plot(X, f(X), label="Fit without Errors")
plt.scatter(length, mean_time, color="red", label="Real Data Points")
plt.grid(True)
plt.xlabel("Length (cm)")
plt.ylabel("Mean Time Period (T²)")
plt.title("Fit for Mean Time Period vs Length without Error")

# Fit With Errors
coefficients, cov = np.polyfit(length, STD_Time_Period, 1, w = (1/STD_Time_Period))
plt.legend()
plt.show()