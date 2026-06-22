# Algorithm Name : Support Vector Machine (SVM) Algorithm
# Video          : Mastering Support Vector Machines with Python and Scikit-Learn
# Source         : https://youtu.be/kPkwf1x7zpU?si=VGVf8P1iY5ea1pSV

# Import Pandas for data manipulation and analysis
import pandas as pd

# Import NumPy for numerical computations and random data generation
import numpy as np

# Import Matplotlib for data visualization
import matplotlib.pyplot as plt

# Import function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import Support Vector Classifier (SVM model)
from sklearn.svm import SVC

# Define the mean, standard deviation and number of samples for the Miles_Per_week feature
mean1 = 55
std_dev1 = 10
num_samples = 500

# Generate random values following a normal distribution
column1_numbers = np.random.normal(mean1, std_dev1, num_samples)

# Limit the values to the range 30 to 120
column1_numbers = np.clip(column1_numbers, 30, 120)

# Round values and convert them to integers
column1_numbers = np.round(column1_numbers).astype(int)

# Define the mean and standard deviation for the Farthest_run feature
mean2 = 18
std_dev2 = 3

# Generate random values following a normal distribution
column2_numbers = np.random.normal(mean2, std_dev2, num_samples)

# Limit the values to the range 12 to 26
column2_numbers = np.clip(column2_numbers, 12, 26)

# Round values and convert them to integers
column2_numbers = np.round(column2_numbers).astype(int)

# Generate random binary values (0 or 1) representing qualification status
column3_numbers = np.random.randint(2, size=num_samples)

# Set qualification to 1 when Miles_Per_week is greater than the mean
column3_numbers[column1_numbers > mean1] = 1

# Create a dictionary containing all columns
data = {
    'Miles_Per_week': column1_numbers,
    'Farthest_run': column2_numbers,
    'Qualified_Boston_Marathon': column3_numbers
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Display the generated dataset
print(df)

# Create a scatter plot to visualize the data
plt.figure(figsize=(10, 6))

# Plot Miles_Per_week against Farthest_run
# Color points according to qualification status
plt.scatter(
    df['Miles_Per_week'],
    df['Farthest_run'],
    c=df['Qualified_Boston_Marathon'],
    cmap='coolwarm'
)

# Label the x-axis
plt.xlabel('Miles Per Week')

# Label the y-axis
plt.ylabel('Farthest Run')

# Add a title to the plot
plt.title('Scatter Plot')

# Display the color scale legend
plt.colorbar(label='Qualified Boston Marathon')

# Show the plot
plt.show()

# Select the first two columns as input features
X = df.iloc[:, 0:2]
print(X)

# Select the third column as the target variable
y = df.iloc[:, 2]
print(y)

# Split the dataset into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)

# Create an SVM model with default parameters
model = SVC()

# Train the model using the training data
model.fit(X_train, y_train)

# Display the model accuracy on the test dataset
print(model.score(X_test, y_test))

# Regularization Parameter (C)
# Smaller C allows more classification errors
# and creates a wider margin
model_reg0 = SVC(C=0.1)

# Train the model
model_reg0.fit(X_train, y_train)

# Display the model accuracy
print(model_reg0.score(X_test, y_test))

# Create an SVM model with the default C value
model_reg1 = SVC(C=1)

# Train the model
model_reg1.fit(X_train, y_train)

# Display the model accuracy
print(model_reg1.score(X_test, y_test))

# Create an SVM model with a large C value
# Larger C attempts to classify all training points correctly
model_reg2 = SVC(C=1000)

# Train the model
model_reg2.fit(X_train, y_train)

# Display the model accuracy
print(model_reg2.score(X_test, y_test))

# Gamma Parameter
# Smaller gamma creates a smoother decision boundary
model_gamma0 = SVC(gamma=0.1)

# Train the model
model_gamma0.fit(X_train, y_train)

# Display the model accuracy
print(model_gamma0.score(X_test, y_test))

# Gamma Parameter
# Medium gamma value
model_gamma1 = SVC(gamma=1)

# Train the model
model_gamma1.fit(X_train, y_train)

# Display the model accuracy
print(model_gamma1.score(X_test, y_test))

# Gamma Parameter
# Very large gamma may lead to overfitting
model_gamma2 = SVC(gamma=1000)

# Train the model
model_gamma2.fit(X_train, y_train)

# Display the model accuracy
print(model_gamma2.score(X_test, y_test))

# Kernel Function
# Use a linear decision boundary
model_linear = SVC(kernel='linear')

# Train the model
model_linear.fit(X_train, y_train)

# Display the model accuracy
print(model_linear.score(X_test, y_test))

# Kernel Function
# Use the Radial Basis Function (RBF) kernel
model_linear = SVC(kernel='rbf')

# Train the model
model_linear.fit(X_train, y_train)

# Display the model accuracy
print(model_linear.score(X_test, y_test))