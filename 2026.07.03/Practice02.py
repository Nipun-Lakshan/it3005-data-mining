# Name                : A. W. W. A. Nipun Lakshan
# Registration Number : 2023s20371
# Index Number        : s17618
# Assignment          : Identification of Optimal Spectral Wavelengths for Classifying Sri Lankan Tea Origins Using Random Forest and Support Vector Machine

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

# Import Median Filter Library
from scipy.signal import medfilt

# Import Gaussian Filter Library
from scipy.ndimage import gaussian_filter1d

# Step 01: Data Collection, Loading and Exploration
# =================================================

# Define a List of the Y Columns from the First 16 Datasets of Dimbula
Y_D1 = []

# Define a List of the Average Y Column from the First 16 Datasets of Dimbula
Y_AD1 = []

# Define a List of the Y Columns from the Second 16 Datasets of Dimbula
Y_D2 = []

# Define a List of the Average Y Column from the Second 16 Datasets of Dimbula
Y_AD2 = []

# Define a List of the Y Columns from the Third 16 Datasets of Dimbula
Y_D3 = []

# Define a List of the Average Y Column from the Third 16 Datasets of Dimbula
Y_AD3 = []

# Define the Wave Length List
data = "Tea Data\\Dimbula\\D1.csv"
df = pd.read_csv(data)
X = df['X'].tolist()

# Run a Loop to Fill the Y_D1 List
for i in range(1, 17):
    data = "Tea Data\\Dimbula\\D" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_D1.append(list(df['Y']))

# Run a Loop to Fill the Y_D2 List
for i in range(17, 33):
    data = "Tea Data\\Dimbula\\D" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_D2.append(list(df['Y']))

# Run a Loop to Fill the Y_D3 List
for i in range(33, 49):
    data = "Tea Data\\Dimbula\\D" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_D3.append(list(df['Y']))

# Run a Loop to Create Y_AD1 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_D1[j][i]
    Y_AD1.append((total / 16))

# Run a Loop to Create Y_AD2 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_D2[j][i]
    Y_AD2.append((total / 16))

# Run a Loop to Create Y_AD3 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_D3[j][i]
    Y_AD3.append((total / 16))

# Plot the Three Average Datasets in One Plot
plt.figure()
plt.title("Average of the First, Second & Third 16 Datasets for Dimbula Tea")
plt.plot(X, Y_AD1, color="blue", label="Average of First 16")
plt.plot(X, Y_AD2, color="red", label="Average of Second 16")
plt.plot(X, Y_AD3, color="green", label="Average of Third 16")
plt.xlabel("Wave Length (nm) - Features")
plt.ylabel("Average Intensity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Define a List of the Y Columns from the First 16 Datasets of Nuwara Eliya
Y_N1 = []

# Define a List of the Average Y Column from the First 16 Datasets of Nuwara Eliya
Y_AN1 = []

# Define a List of the Y Columns from the Second 16 Datasets of Nuwara Eliya
Y_N2 = []

# Define a List of the Average Y Column from the Second 16 Datasets of Nuwara Eliya
Y_AN2 = []

# Define a List of the Y Columns from the Third 16 Datasets of Nuwara Eliya
Y_N3 = []

# Define a List of the Average Y Column from the Third 16 Datasets of Nuwara Eliya
Y_AN3 = []

# Run a Loop to Fill the Y_N1 List
for i in range(1, 17):
    data = "Tea Data\\Nuwara Eliya\\N" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_N1.append(list(df['Y']))

# Run a Loop to Fill the Y_N2 List
for i in range(17, 33):
    data = "Tea Data\\Nuwara Eliya\\N" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_N2.append(list(df['Y']))

# Run a Loop to Fill the Y_N3 List
for i in range(33, 49):
    data = "Tea Data\\Nuwara Eliya\\N" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_N3.append(list(df['Y']))

# Run a Loop to Create Y_AN1 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_N1[j][i]
    Y_AN1.append((total / 16))

# Run a Loop to Create Y_AN2 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_N2[j][i]
    Y_AN2.append((total / 16))

# Run a Loop to Create Y_AN3 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_N3[j][i]
    Y_AN3.append((total / 16))

# Plot the Three Average Datasets in One Plot
plt.figure()
plt.title("Average of the First, Second & Third 16 Datasets for Nuwara Eliya Tea")
plt.plot(X, Y_AN1, color="blue", label="First 16 Average")
plt.plot(X, Y_AN2, color="red", label="Second 16 Average")
plt.plot(X, Y_AN3, color="green", label="Third 16 Average")
plt.xlabel("Wave Length (nm) - Features")
plt.ylabel("Average Intensity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Define a List of the Y Columns from the First 16 Datasets of Uva
Y_U1 = []

# Define a List of the Average Y Column from the First 16 Datasets of Uva
Y_AU1 = []

# Define a List of the Y Columns from the Second 16 Datasets of Uva
Y_U2 = []

# Define a List of the Average Y Column from the Second 16 Datasets of Uva
Y_AU2 = []

# Define a List of the Y Columns from the Third 16 Datasets of Uva
Y_U3 = []

# Define a List of the Average Y Column from the Third 16 Datasets of Uva
Y_AU3 = []

# Run a Loop to Fill the Y_U1 List
for i in range(1, 17):
    data = "Tea Data\\Uva\\U" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_U1.append(list(df['Y']))

# Run a Loop to Fill the Y_U2 List
for i in range(17, 33):
    data = "Tea Data\\Uva\\U" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_U2.append(list(df['Y']))

# Run a Loop to Fill the Y_U3 List
for i in range(33, 49):
    data = "Tea Data\\Uva\\U" + str(i) + ".csv"
    df = pd.read_csv(data)
    Y_U3.append(list(df['Y']))

# Run a Loop to Create Y_AU1 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_U1[j][i]
    Y_AU1.append((total / 16))

# Run a Loop to Create Y_AU2 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_U2[j][i]
    Y_AU2.append((total / 16))

# Run a Loop to Create Y_AU3 List
for i in range(0, 3200):
    total = 0
    for j in range(0, 16):
        total += Y_U3[j][i]
    Y_AU3.append((total / 16))

# Plot the Three Average Datasets in One Plot
plt.figure()
plt.title("Average of the First, Second & Third 16 Datasets for Uva Tea")
plt.plot(X, Y_AU1, color="blue", label="First 16 Average")
plt.plot(X, Y_AU2, color="red", label="Second 16 Average")
plt.plot(X, Y_AU3, color="green", label="Third 16 Average")
plt.xlabel("Wave Length (nm) - Features")
plt.ylabel("Average Intensity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Step 02: Filter the Noise of the Spectrum
# =========================================

# Salt (High Intensity) and Pepper (Low Intensity) Noise - Type of Noise

# Note
# ====

# Noise spikes should be removed before analysis and ML training because they can
# create false patterns, reduce feature accuracy, and decrease model performance.

# Part 01: Using Moving Average
# =============================

# Prepare X Values for MA-7
X_MA7 = X[0:-6]

# Dimbula - First 16 Datasets
Y_MD1 = []

# Run a Loop to Fill the MD1 List
for i in range(0, (len(Y_AD1) - 6)):
    Y_MD1.append((float(Y_AD1[i]) + float(Y_AD1[i + 1]) + float(Y_AD1[i + 2]) + float(Y_AD1[i + 3]) + float(
        Y_AD1[i + 4]) + float(Y_AD1[i + 5]) + float(Y_AD1[i + 6])) / 7)

# Dimbula - Second 16 Datasets
Y_MD2 = []

# Run a Loop to Fill the MD2 List
for i in range(0, (len(Y_AD2) - 6)):
    Y_MD2.append((float(Y_AD2[i]) + float(Y_AD2[i + 1]) + float(Y_AD2[i + 2]) + float(Y_AD2[i + 3]) + float(
        Y_AD2[i + 4]) + float(Y_AD2[i + 5]) + float(Y_AD2[i + 6])) / 7)

# Dimbula - Third 16 Datasets
Y_MD3 = []

# Run a Loop to Fill the MD3 List
for i in range(0, (len(Y_AD3) - 6)):
    Y_MD3.append((float(Y_AD3[i]) + float(Y_AD3[i + 1]) + float(Y_AD3[i + 2]) + float(Y_AD3[i + 3]) + float(
        Y_AD3[i + 4]) + float(Y_AD3[i + 5]) + float(Y_AD3[i + 6])) / 7)

# Note
# ====

# MA-3  → Light smoothing
# MA-5  → Moderate smoothing
# MA-7 → Strong smoothing with possible peak distortion
# MA-9 or MA-11 → Very strong smoothing, but may distort spectral peaks

# Plot the Graph for Three Datasets from Dimbula Using a 7-Point Moving Average
plt.figure()
plt.title("MA-7 Plot for Three Dimbula Datasets")
plt.plot(X_MA7, Y_MD1, color="blue", label="Dimbula - D1 - MA7")
plt.plot(X_MA7, Y_MD2, color="red", label="Dimbula - D2 - MA7")
plt.plot(X_MA7, Y_MD3, color="green", label="Dimbula - D3 - MA7")
plt.xlabel("Wave Length (nm) - Features [For MA-7]")
plt.ylabel("Average Intensity (MA-7)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Note
# ====

# MA-3, MA-5 and MA-7 are not sufficient for effective noise removal. If MA-7 is unable to
# remove the noise, MA-3 and MA-5 will also be insufficient for effective noise reduction.

# Part 02: Applying Median Filter
# ===============================

# Note
# ====

# Kernel Size = The number of data points included
# the moving window to calculate the median value.

# For Three Dimbula Datasets
Y_MF1 = medfilt(Y_AD1, kernel_size=5)
Y_MF2 = medfilt(Y_AD2, kernel_size=5)
Y_MF3 = medfilt(Y_AD3, kernel_size=5)

# Plot the Graph for Three Datasets from Dimbula Using Median Filter
plt.figure()
plt.title("Median Filter Plot for Three Dimbula Datasets")
plt.plot(X, Y_MF1, color="blue", label="Dimbula - D1 - MF5")
plt.plot(X, Y_MF2, color="red", label="Dimbula - D2 - MF5")
plt.plot(X, Y_MF3, color="green", label="Dimbula - D3 - MF5")
plt.xlabel("Wave Length (nm) - Features [For MF]")
plt.ylabel("Average Intensity (MF)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Part 03: Applying Gaussian Filter
# =================================

# Note
# ====

# Gaussian Filter reduces random noise and produces a smoother
# spectrum while maintaining the general signal pattern.

# sigma (σ) - Parameter in the Gaussian Filter Function

# Small (0.5 - 1) -> Light smoothing
# Medium (1 - 2)  -> Moderate smoothing
# Large (>2)	  -> Strong smoothing

# Smaller sigma values produce less smoothing.
# Larger sigma values produce stronger smoothing.

# For Three Dimbula Datasets [For Sigma = 1]
Y_GF1 = gaussian_filter1d(Y_AD1, sigma=1)
Y_GF2 = gaussian_filter1d(Y_AD2, sigma=1)
Y_GF3 = gaussian_filter1d(Y_AD3, sigma=1)

# Plot the Graph for Three Datasets from Dimbula Using a Gaussian Filter
plt.figure()
plt.title("Gaussian Filter Plot for Three Dimbula Datasets (Sigma=1)")
plt.plot(X, Y_GF1, color="blue", label="Dimbula - D1 - GF")
plt.plot(X, Y_GF2, color="red", label="Dimbula - D2 - GF")
plt.plot(X, Y_GF3, color="green", label="Dimbula - D3 - GF")
plt.xlabel("Wave Length (nm) - Features [Gaussian Filter]")
plt.ylabel("Average Intensity (G)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# For Three Dimbula Datasets [For Sigma = 20]
Y_GFD1 = gaussian_filter1d(Y_AD1, sigma=20)
Y_GFD2 = gaussian_filter1d(Y_AD2, sigma=20)
Y_GFD3 = gaussian_filter1d(Y_AD3, sigma=20)

# Plot the Graph for Three Datasets from Dimbula Using a Gaussian Filter
plt.figure()
plt.title("Gaussian Filter Plot for Three Dimbula Datasets (Sigma=20)")
plt.plot(X, Y_GFD1, color="blue", label="Dimbula - D1 - GF20")
plt.plot(X, Y_GFD2, color="red", label="Dimbula - D2 - GF20")
plt.plot(X, Y_GFD3, color="green", label="Dimbula - D3 - GF20")
plt.xlabel("Wave Length (nm) - Features [Gaussian Filter]")
plt.ylabel("Average Intensity (G)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# For Three Nuwara Eliya Datasets [For Sigma = 20]
Y_GFN1 = gaussian_filter1d(Y_AN1, sigma=20)
Y_GFN2 = gaussian_filter1d(Y_AN2, sigma=20)
Y_GFN3 = gaussian_filter1d(Y_AN3, sigma=20)

# Plot the Graph for Three Datasets from Nuwara Eliya Using a Gaussian Filter
plt.figure()
plt.title("Gaussian Filter Plot for Three Nuwara Eliya Datasets (Sigma=20)")
plt.plot(X, Y_GFN1, color="blue", label="Nuwara Eliya - N1 - GF20")
plt.plot(X, Y_GFN2, color="red", label="Nuwara Eliya - N2 - GF20")
plt.plot(X, Y_GFN3, color="green", label="Nuwara Eliya - N3 - GF20")
plt.xlabel("Wave Length (nm) - Features [Gaussian Filter]")
plt.ylabel("Average Intensity (G)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# For Three Uva Datasets [For Sigma = 20]
Y_GFU1 = gaussian_filter1d(Y_AU1, sigma=20)
Y_GFU2 = gaussian_filter1d(Y_AU2, sigma=20)
Y_GFU3 = gaussian_filter1d(Y_AU3, sigma=20)

# Plot the Graph for Three Datasets from Uva Using a Gaussian Filter
plt.figure()
plt.title("Gaussian Filter Plot for Three Uva Datasets (Sigma=20)")
plt.plot(X, Y_GFU1, color="blue", label="Uva - U1 - GF20")
plt.plot(X, Y_GFU2, color="red", label="Uva - U2 - GF20")
plt.plot(X, Y_GFU3, color="green", label="Uva - U3 - GF20")
plt.xlabel("Wave Length (nm) - Features [Gaussian Filter]")
plt.ylabel("Average Intensity (G)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Note
# ====

# Sigma (σ) is the standard deviation of the Gaussian distribution used in the Gaussian filter.
# It controls the amount of smoothing applied to the spectrum. A larger σ produces stronger
# smoothing, while a smaller σ preserves more spectral details.

# Step 03: Creating a Suitable DataFrame for Supervised Machine Learning Models
# =============================================================================

# Convert Float List into String List
Feature_List = [str(x) for x in X]
Feature_List.append("Tea Types")

# Create an empty dataset
data = pd.DataFrame(columns=Feature_List)

# Adding Rows of Dimbula Data
for i in range(1, 51):
    data1 = "Tea Data\\Dimbula\\D" + str(i) + ".csv"
    df = pd.read_csv(data1)
    data.loc[(len(data))] = gaussian_filter1d(list(df['Y']), sigma=20).tolist() + ["Dimbula"]

# Adding Rows of Nuwara Eliya Data
for i in range(1, 51):
    data1 = "Tea Data\\Nuwara Eliya\\N" + str(i) + ".csv"
    df = pd.read_csv(data1)
    data.loc[(len(data))] = gaussian_filter1d(list(df['Y']), sigma=20).tolist() + ["Nuwara Eliya"]

# Adding Rows of Uva Data
for i in range(1, 51):
    data1 = "Tea Data\\Uva\\U" + str(i) + ".csv"
    df = pd.read_csv(data1)
    data.loc[(len(data))] = gaussian_filter1d(list(df['Y']), sigma=20).tolist() + ["Uva"]

# Step 04: Identification of the Best Features Using Random Forest Classifier
# ===========================================================================

# Defining the X and y Lists [axis=0 -> Rows / axis=1 -> Columns]
Xr = data.drop('Tea Types', axis=1)
Yr = data['Tea Types']

# Split the Dataset [stratify=Yr => To ensure that the training and testing datasets are balanced.]
X_train, X_test, Y_train, Y_test = train_test_split(Xr, Yr, test_size=0.3, random_state=42, stratify=Yr)

# Create Random Forest Model
RF = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the Model
RF.fit(X_train, Y_train)

# Make Predictions
Y_pred = RF.predict(X_test)

# Print Header
print("\n===========================")
print("01. Random Forest Technique")
print("===========================\n")

# Evaluate the Model
print("Accuracy of Random Forest:", round((accuracy_score(Y_test, Y_pred) * 100), 2), "\n")

# Create confusion matrix graph
ConfusionMatrixDisplay.from_predictions(
    Y_test,
    Y_pred
)
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Get Feature Importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": RF.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# Header
print("=====================================")
print("Best Two Features - From RF Algorithm")
print("=====================================\n")

# Best Two Features
print(importance.head(2))

# Header
print("\n=====================")
print("Classification Report")
print("=====================\n")

print(classification_report(Y_test, Y_pred))

# Precision : When the model predicts a class, how often is it correct?
# Recall    : From all actual samples of a class, how many did the model correctly find?
# F1-Score  : The balance between precision and recall.
# Support   : Number of actual samples belonging to each class in the test dataset.

# Step 05: Identification of the Best Features Using SVM Classifier
# =================================================================

# Print Header
print("\n=================")
print("02. SVM Technique")
print("=================\n")

# Train SVM Machine
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, Y_train)

# Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": abs(svm.coef_).mean(axis=0)
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# Train the Model
svm.fit(X_train, Y_train)

# Make Predictions
Y_pred = svm.predict(X_test)

# Evaluate the Model
print("Accuracy of SVM Technique:", round((accuracy_score(Y_test, Y_pred) * 100), 2), "\n")

# Header
print("======================================")
print("Best Two Features - From SVM Algorithm")
print("======================================\n")

print(importance.head(2))

# Header
print("\n=====================")
print("Classification Report")
print("=====================\n")

print(classification_report(Y_test, Y_pred))

# Create confusion matrix graph
ConfusionMatrixDisplay.from_predictions(
    Y_test,
    Y_pred
)
plt.title("Confusion Matrix - SVM")
plt.show()