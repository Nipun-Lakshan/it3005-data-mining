# Algorithm Name : Random Forest Algorithm
# Video          : Random Forest Algorithm Explained with Python and scikit-learn
# Source         : https://youtu.be/_QuGM_FW9eo?si=4MR6pA2JcCQdcpd4

# Import Pandas for data manipulation and analysis
import pandas as pd

# Import function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import the Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier

# Import function to generate the classification report
from sklearn.metrics import classification_report

# Load the dataset from the CSV file
df = pd.read_csv('500hits.csv', encoding='latin-1')

# Display the first 5 rows of the dataset
print(df.head())

# Remove unwanted columns from the dataset
df = df.drop(columns=['PLAYER', 'CS'])

# Select all rows and the first 13 columns as input features
X = df.iloc[:, 0:13]

# Select all rows from the 14th column as the target variable
y = df.iloc[:, 13]

# Split the dataset into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

# Create a Random Forest Classifier with default parameters
rf = RandomForestClassifier()

# Train the Random Forest model using the training data
rf.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = rf.predict(X_test)

# Display the model accuracy on the test dataset
print(rf.score(X_test, y_test))

# Display precision, recall, F1-score, and support for each class
print(classification_report(y_test, y_pred))

# Create a DataFrame to display feature importance values
features = pd.DataFrame(rf.feature_importances_, index=X.columns)

# Display the feature importance table
print(features.head(15))

# Hyperparameters

# Create a Random Forest Classifier with custom hyperparameters
# n_estimators      : Number of decision trees in the forest
# criterion         : Measure used to split nodes ('entropy' uses information gain)
# min_samples_split : Minimum number of samples required to split a node
# max_depth         : Maximum depth of each tree
# random_state      : Ensures reproducible results
rf2 = RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    min_samples_split=10,
    max_depth=14,
    random_state=42
)

# Train the tuned Random Forest model
rf2.fit(X_train, y_train)

# Display the accuracy of the tuned model
print(rf2.score(X_test, y_test))

# Predict class labels using the tuned model
y_pred2 = rf2.predict(X_test)

# Display the classification report of the tuned model
print(classification_report(y_test, y_pred2))