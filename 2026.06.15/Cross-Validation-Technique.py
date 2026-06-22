# Technique Name : Cross - Validation Technique
# Video          : Machine Learning Tutorial Python 12 - K-Fold Cross Validation
# Source         : https://youtu.be/gJo0uNL-5Qw?si=wRDY9h5iS5o6AMJT

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Import Support Vector Classifier model
from sklearn.svm import SVC

# Import Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier

# Import K-Fold cross-validation technique
from sklearn.model_selection import KFold

# Load handwritten digits dataset
from sklearn.datasets import load_digits

# Import Stratified K-Fold cross-validation (maintains class distribution)
from sklearn.model_selection import StratifiedKFold

# Import function for evaluating models using cross-validation
from sklearn.model_selection import cross_val_score

# Load dataset into variable
digits = load_digits()

# Split dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# Create Logistic Regression model
lr = LogisticRegression(max_iter=500)

# Train Logistic Regression model
lr.fit(X_train, y_train)

# Evaluate Logistic Regression model accuracy
print(lr.score(X_test, y_test))

# Create Support Vector Machine model
svm = SVC()

# Train SVM model
svm.fit(X_train, y_train)

# Evaluate SVM model accuracy
print(svm.score(X_test, y_test))

# Create Random Forest model with 40 trees
rf = RandomForestClassifier(n_estimators=40)

# Train Random Forest model
rf.fit(X_train, y_train)

# Evaluate Random Forest model accuracy
print(rf.score(X_test, y_test))

# Create K-Fold cross-validator with 3 splits
kf = KFold(n_splits=3)

# Print K-Fold configuration
print(kf)

# Show how indices are split in K-Fold
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]): print(train_index, test_index)

# Function to train a model and return its accuracy
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Evaluate Logistic Regression using helper function
print(get_score(LogisticRegression(max_iter=500), X_train, X_test, y_train, y_test))

# Evaluate SVM using helper function
print(get_score(SVC(), X_train, X_test, y_train, y_test))

# Create Stratified K-Fold cross-validator with 3 splits
folds = StratifiedKFold(n_splits=3)

# Lists to store model scores for each fold
scores_l = []
scores_SVM = []
scores_rf = []

# Perform K-Fold cross-validation manually
for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scores_l.append(get_score(LogisticRegression(max_iter=500), X_train, X_test, y_train, y_test))
    scores_SVM.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

# Print results of manual cross-validation for each model
print("LR  : ", scores_l)
print("SVM : ", scores_SVM)
print("RF  : ", scores_rf)

# Perform cross-validation using built-in function (Logistic Regression)
print(cross_val_score(LogisticRegression(max_iter=500), digits.data, digits.target))

# Perform cross-validation using built-in function (SVM)
print(cross_val_score(SVC(), digits.data, digits.target))

# Parameter tuning example (changing number of trees in Random Forest)
print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target))