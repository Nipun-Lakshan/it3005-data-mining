# Algorithm Name : Decision Tree Algorithm [For Three-Labeled Data Points]
# Video          : How to Build Your First Decision Tree in Python (scikit-learn)
# Source         : https://youtu.be/YkYpGhsCx4c?si=Ad2KPtf9__dy9-Wh

# Note

"""
Decision Trees support both binary and multiclass classification. There is no theoretical maximum
number of class labels, although performance may decrease as the number of classes increases.
"""

# Import Libraries
# ================

# Import Pandas for data manipulation and analysis
import pandas as pd

# Import function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import the Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier

# Import function to generate the confusion matrix
from sklearn.metrics import confusion_matrix

# Import function to generate the classification report
from sklearn.metrics import classification_report

# Load Dataset from CSV File
df = pd.read_csv('500hits.csv', encoding='latin-1')
"""
The encoding parameter tells Pandas how to interpret the bytes in the file as characters.
Without the correct encoding, special characters may appear incorrectly or cause errors.
"""

# Print First 5 Rows
print(df.head())

# Drop Two Columns (Remove unwanted columns)
# PLAYER and CS are removed because they are not used for training
df = df.drop(columns=['PLAYER', 'CS'])

# Select all rows and the first 13 columns of the DataFrame and store them in X
# These columns are used as input features
X = df.iloc[:,0:13] # Columns 0 to 12

# Select all rows from the 14th column and store them in y
# This column contains the target/class label
y = df.iloc[:,13]

# Split the dataset into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

# Print the shape of the training feature set
print(X_train.shape)

# Print the shape of the training target set
print(y_train.shape)

# Print the shape of the testing feature set
print(X_test.shape)

# Print the shape of the testing target set
print(y_test.shape)

# Create a Decision Tree Classifier object with default parameters
dtc = DecisionTreeClassifier()

# Display all parameters of the classifier
print(dtc.get_params())

# Train (fit) the model using the training data
dtc.fit(X_train, y_train)

# Predict class labels for the test data
y_pred = dtc.predict(X_test)

# Display the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Display performance metrics such as precision, recall, and F1-score
print(classification_report(y_test, y_pred))

# Display feature importance scores
print(dtc.feature_importances_)

# Display feature names
print(X.columns)

# Create a DataFrame to show feature importance with feature names
features = pd.DataFrame(dtc.feature_importances_, index=X.columns)
print(features.head(15))

# Create another Decision Tree Classifier using entropy
dtc2 = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.04)

'''CCP stops data overfitting.'''

# Train the Decision Tree model
dtc2.fit(X_train, y_train)

'''
Pruned : Model that has been simplified by removing unnecessary branches from a Decision Tree.
'''
# Predict class labels using the pruned model
y_pred2 = dtc2.predict(X_test)

# Display the confusion matrix for the pruned model
print(confusion_matrix(y_test, y_pred2))

# Display performance metrics for the pruned model
print(classification_report(y_test, y_pred2))

# Create a DataFrame to show feature importance of the pruned model
features2 = pd.DataFrame(dtc2.feature_importances_, index=X.columns)

# Display feature importance values
print(features2.head(15))