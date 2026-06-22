import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 01. Load Dataset
data = pd.read_csv('sensor.csv')
print("\n")

# 02. Plot The Graph
X = data['Temperature(c)']
y1 = data['Voltage (heating)']
y2 = data['Voltage (cooling)']
y = data['mean']

plt.figure()
plt.scatter(X, y1, label='Voltage (Heating)')
plt.scatter(X, y2, label='Voltage (Cooling)')
plt.legend()
plt.grid()
plt.xlabel("Temperature")
plt.ylabel("Voltage Values")
plt.title("Temperature vs Voltage Values")
plt.show()

# 03. RMSE Values
orders = []
rmse_values = []

for order in range(1, 7):
    coefficients = np.polyfit(X, y, order)
    polynomial = np.poly1d(coefficients)
    y_pred = polynomial(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    orders.append(order)
    rmse_values.append(float(rmse))

for index in range(1, 7):
    print(f"RMSE Value For Order {index} : {rmse_values[index - 1]}")

# 04. Plot Order vs RMSE Graph
plt.figure()
plt.scatter(range(1, 7), rmse_values)
plt.plot(range(1, 7), rmse_values)
plt.xlabel("Order")
plt.ylabel("RMSE")
plt.title("RMSE vs Order")
plt.grid()
plt.show()

# 05. Best Fitted Line
best_order = 3
coefficients = np.polyfit(X, y, best_order)
polynomial = np.poly1d(coefficients)
X_smooth = np.linspace(min(X), max(X), 500)
y_smooth = polynomial(X_smooth)
plt.figure()
plt.scatter(X, y, label='Experimental Data')
plt.plot(X_smooth, y_smooth, label=f'Order {best_order} Polynomial Fit')
plt.xlabel('Temperature (°C)')
plt.ylabel('Voltage')
plt.title('Sensor Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()

# 06. Find Unknown Values
print("\n")
orders1 = []
rmse_values1 = []

for order in range(1, 7):
    coefficients = np.polyfit(y, X, order)
    polynomial = np.poly1d(coefficients)
    X_pred = polynomial(y)
    rmse = np.sqrt(mean_squared_error(X, X_pred))
    orders.append(orders1)
    rmse_values1.append(float(rmse))

for index in range(1, 7):
    print(f"RMSE Value For Order {index} : {rmse_values[index - 1]}")

plt.figure()
plt.scatter(range(1, 7), rmse_values1)
plt.plot(range(1, 7), rmse_values1)
plt.xlabel("Order")
plt.ylabel("RMSE")
plt.title("New RMSE vs Order")
plt.grid()
plt.show()

best_order = 3
coefficients = np.polyfit(y, X, best_order)
f = np.poly1d(coefficients)
X_smooth = np.linspace(min(y), max(y), 500)
y_smooth = polynomial(X_smooth)
plt.figure()
plt.scatter(y, X, label='Experimental Data')
plt.plot(X_smooth, y_smooth, label=f'Order {best_order} Polynomial Fit')
plt.ylabel('Temperature (°C)')
plt.xlabel('Voltage')
plt.title('Inverse Function')
plt.legend()
plt.grid(True)
plt.show()

V = [4.15, 3.74, 2.58, 3.53, 2.41, 1.80, 2.23, 2.35]
Temperature_Values = []

for key1 in range(1, 9):
    Temperature_Values.append(float(f(V[key1 - 1])))
print("\n")
print("Temperature Values For Given Voltages : ", Temperature_Values)