# Note 16: RandomizedSearchCV & GridSearchCV
# ==========================================

# Import Libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


# 01. Grid Serach CV with Random Forest
# =====================================

# Create a Model
model = RandomForestClassifier(random_state=42)

# Set Values
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10]
}

# Run Grid Search CV
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit Model
grid.fit(X_train, y_train)

# Print Data
print(grid.best_params_)
print(grid.best_score_)

# 02. Grid Serach CV with SVM
# ===========================

parameters = {
    'C':[1, 10, 100],
    'kernel':['linear', 'poly', 'rbf']
}

gsv = GridSearchCV(svm, parameters)
gsv.fit(X_train, y_train)

print("Best C Value For GSV : ", gsv.best_score_)
print("Best Parameters      : ", gsv.best_params_)

# 03. Randomized Serach CV with SVM
# =================================

parameters = {
    'C':[1, 10, 100],
    'kernel':['linear', 'poly', 'rbf']
}

rsv = RandomizedSearchCV(svm, parameters, n_iter=5)
rsv.fit(X_train, y_train)
print("Best C Value For RSV : ", rsv.best_score_)
print("Best Parameters      : ", rsv.best_params_)

# 04. Randomized Serach CV with Random Forest
# ===========================================

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(
    model,
    param_grid,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X_train, y_train)

print(random_search.best_params_)
print(random_search.best_score_)