# 01. Manual Implementation

import numpy as np
from sklearn.preprocessing import StandardScaler

data = [150, 160, 170, 180, 190]
np_data = np.array(data)

data_scaled = []

mean = (np_data.mean())
std = (np_data.std())

for i in range(len(np_data)):
    data_scaled.append((np_data[i] - mean) / std)

print("Scaled Data 01 : ", np.array(data_scaled))

# 02. Using Standard Scalar Function

data = [150, 160, 170, 180, 190]
data = np.array(data).reshape(-1, 1)
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print("Scaled Data 02 : ", scaled.reshape(1, -1))

# 03. Without Using Fit Transform

data = [150, 160, 170, 180, 190]
data = np.array(data).reshape(-1, 1)
scaler = StandardScaler()
scaled = scaler.fit(data)
# print(scaler.mean_)
# print(scaler.scale_)
print(scaler.transform(data).reshape(1, -1))

# 04. Exercise

data = np.array([10, 20, 30, 40, 50, 60])
print(data.mean())
print(data.std())

data = np.array([70, 80, 90])
print(data.mean())
print(data.std())