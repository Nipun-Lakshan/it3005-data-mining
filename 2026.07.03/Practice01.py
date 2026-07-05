# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# 01. Load dataset
iris = pd.read_csv("iris.csv")

# 02. Features and target
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

# 03. Train Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 04. Load SVM Algorithm
svm = SVC(kernel='linear')

# 05. Train model
svm.fit(X_train, y_train)

# 06. Predict Model
y_pred = svm.predict(X_test)

# 07. Grid Search CV
parameters = {'C':[1, 10, 100], 'kernel':['linear', 'poly', 'rbf']}
gsv = GridSearchCV(svm, parameters)
gsv.fit(X_train, y_train)
print("Best C Value For GSV : ", gsv.best_score_)
print("Best Parameters      : ", gsv.best_params_)

# 08. Random Search CV
parameters = {'C':[1, 10, 100], 'kernel':['linear', 'poly', 'rbf']}
rsv = RandomizedSearchCV(svm, parameters, n_iter=5)
rsv.fit(X_train, y_train)
print("Best C Value For RSV : ", rsv.best_score_)
print("Best Parameters      : ", rsv.best_params_)