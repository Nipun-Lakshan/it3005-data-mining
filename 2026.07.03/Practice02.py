

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# 01. Create a List Containing the First 16 Datasets of Dimbula
X_D1 = []
Y_D1 = []

# 02. Create a List Containing the Second 16 Datasets of Dimbula
X_D2 = []
Y_D2 = []

# 03. Create a List Containing the Third 16 Datasets of Dimbula
X_D3 = []
Y_D3 = []

# 04. Creating Loop For First 16 Datasets
for key in range(1, 17):
    data = "Tea Data\\Dimbula\\D" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_D1.extend(list(x['X']))
    Y_D1.extend(list(x['Y']))

# 05. Creating Loop For Second 16 Datasets
for key in range(17, 33):
    data = "Tea Data\\Dimbula\\D" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_D2.extend(list(x['X']))
    Y_D2.extend(list(x['Y']))

# 06. Creating Loop For Third 16 Datasets
for key in range(33, 49):
    data = "Tea Data\\Dimbula\\D" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_D3.extend(list(x['X']))
    Y_D3.extend(list(x['Y']))

# 07. Create a List Containing the First 16 Datasets of Nuwara Eliya
X_N1 = []
Y_N1 = []

# 08. Create a List Containing the Second 16 Datasets of Nuwara Eliya
X_N2 = []
Y_N2 = []

# 09. Create a List Containing the Third 16 Datasets of Nuwara Eliya
X_N3 = []
Y_N3 = []

# 10. Creating Loop For First 16 Datasets
for key in range(1, 17):
    data = "Tea Data\\Nuwara Eliya\\N" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_N1.extend(list(x['X']))
    Y_N1.extend(list(x['Y']))

# 11. Creating Loop For Second 16 Datasets
for key in range(17, 33):
    data = "Tea Data\\Nuwara Eliya\\N" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_N2.extend(list(x['X']))
    Y_N2.extend(list(x['Y']))

# 12. Creating Loop For Third 16 Datasets
for key in range(33, 49):
    data = "Tea Data\\Nuwara Eliya\\N" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_N3.extend(list(x['X']))
    Y_N3.extend(list(x['Y']))

# 13. Create a List Containing the First 16 Datasets of Uva
X_U1 = []
Y_U1 = []

# 14. Create a List Containing the Second 16 Datasets of Uva
X_U2 = []
Y_U2 = []

# 15. Create a List Containing the Third 16 Datasets of Uva
X_U3 = []
Y_U3 = []

# 16. Creating Loop For First 16 Datasets
for key in range(1, 17):
    data = "Tea Data\\Uva\\U" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_U1.extend(list(x['X']))
    Y_U1.extend(list(x['Y']))

# 17. Creating Loop For Second 16 Datasets
for key in range(17, 33):
    data = "Tea Data\\Uva\\U" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_U2.extend(list(x['X']))
    Y_U2.extend(list(x['Y']))

# 18. Creating Loop For Third 16 Datasets
for key in range(33, 49):
    data = "Tea Data\\Uva\\U" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_U3.extend(list(x['X']))
    Y_U3.extend(list(x['Y']))

# 19. Draw Nine Plots for the Three Datasets Created for Each Area from the 48 Datasets in Each Area
X = [X_D1, X_D2, X_D3, X_N1, X_N2, X_N3, X_U1, X_U2, X_U3]
Y = [Y_D1, Y_D2, Y_D3, Y_N1, Y_N2, Y_N3, Y_U1, Y_U2, Y_U3]
X_Labels = ["D1X", "D2X", "D3X", "N1X", "N2X", "N3X", "U1X", "U2X", "U3X"]
Y_Labels = ["D1Y", "D2Y", "D3Y", "N1Y", "N2Y", "N3Y", "U1Y", "U2Y", "U3Y"]

for key in range(0, 9):
    plt.figure()
    plt.plot(X[key], Y[key])
    plt.xlabel(X_Labels[key])
    plt.ylabel(Y_Labels[key])
    temp_string = X_Labels[key] + " vs " + Y_Labels[key]
    plt.title(temp_string)
    plt.show()

# 20. Plot a Graph For Whole Dataset
X_ALL = []
Y_ALL = []

for key in range(0, 9):
    X_ALL.extend(X[key])
    Y_ALL.extend(Y[key])

for key in [49, 50]:
    data = "Tea Data\\Dimbula\\D" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_ALL.extend(list(x['X']))
    Y_ALL.extend(list(x['Y']))

for key in [49, 50]:
    data = "Tea Data\\Nuwara Eliya\\N" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_ALL.extend(list(x['X']))
    Y_ALL.extend(list(x['Y']))

for key in [49, 50]:
    data = "Tea Data\\Uva\\U" + str(key) +  ".csv"
    x = pd.read_csv(data)
    X_ALL.extend(list(x['X']))
    Y_ALL.extend(list(x['Y']))

plt.figure()
plt.plot(X_ALL, Y_ALL)
plt.xlabel("Wave Lengths (nm) - For 150 Datasets")
plt.ylabel("Intensity - For 150 Datasets")
plt.title("Wave Lengths and Intensity - For 150 Datasets")
plt.show()

# 21. Calculate the Moving Average and New Lists
X_D1_MA3 = []
X_D2_MA3 = []
X_D3_MA3 = []

X_N1_MA3 = []
X_N2_MA3 = []
X_N3_MA3 = []

X_U1_MA3 = []
X_U2_MA3 = []
X_U3_MA3 = []

Y_D1_MA3 = []
Y_D2_MA3 = []
Y_D3_MA3 = []

Y_N1_MA3 = []
Y_N2_MA3 = []
Y_N3_MA3 = []

Y_U1_MA3 = []
Y_U2_MA3 = []
Y_U3_MA3 = []

MA3_Lists = [
    X_D1_MA3, X_D2_MA3, X_D3_MA3,
    X_N1_MA3, X_N2_MA3, X_N3_MA3,
    X_U1_MA3, X_U2_MA3, X_U3_MA3,
    Y_D1_MA3, Y_D2_MA3, Y_D3_MA3,
    Y_N1_MA3, Y_N2_MA3, Y_N3_MA3,
    Y_U1_MA3, Y_U2_MA3, Y_U3_MA3
]

All_Lists = [
    X_D1, X_D2, X_D3,
    X_N1, X_N2, X_N3,
    X_U1, X_U2, X_U3,
    Y_D1, Y_D2, Y_D3,
    Y_N1, Y_N2, Y_N3,
    Y_U1, Y_U2, Y_U3
]

for i in range(0, len(All_Lists)):
    for j in range(0, (len(All_Lists[i]) - 2)):
        MA3_Lists[i].append((All_Lists[i][j] + All_Lists[i][j + 1] + All_Lists[i][j + 2]) / 3)
