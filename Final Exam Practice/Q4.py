# =========================================
# IT 3005 - Final Exam - Practical Question
# =========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

# ====================
# Part a: Load Dataset
# ====================
df = pd.read_csv('data_set_Q4.csv')

# ===========================
# Part b: Display the dataset
# ===========================
print("\n===============")
print("Display Dataset")
print("===============\n")
print(df.head())

# ==============
# Define X and y
# ==============
X = df.drop("Category", axis=1)
y = df["Category"]

# ==================
# Part c - i: 2 PCAs
# ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
plt.figure()
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1],)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA 2 For Dataset")
plt.grid(True, alpha=0.5)
plt.show()

# ===================
# Part c - ii: 3 PCAs
# ===================
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x = X_pca_3[:, 0]
Y = X_pca_3[:, 1]
z = X_pca_3[:, 2]
ax.scatter(x, Y, z)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
plt.title("PCA 3 For Dataset")
ax.grid(True, alpha=0.3)
plt.show()

# ======================================
# Part d - i: SVM using 2 PCA Components
# ======================================

# Print Header
print("\n==============")
print("PCA 2 Analysis")
print("==============")

# Model Training
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_pca_2, y, test_size=0.3, random_state=42, stratify=y)
svm_2 = SVC(kernel="rbf", random_state=42)
cv_scores_2 = cross_val_score(svm_2, X_train_2, y_train_2, cv=5)
print("\nCross Validation Score:", cv_scores_2)
print("Mean CV Accuracy      :", round((cv_scores_2.mean() * 100), 2), "\b%")
svm_2.fit(X_train_2, y_train_2)
y_pred_2 = svm_2.predict(X_test_2)

# ==========================
# Part d - ii: SVM for PCA 3
# ==========================

# Print Header
print("\n==============")
print("PCA 3 Analysis")
print("==============")

# Model Training
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_pca_3, y, test_size=0.3, random_state=42)
svm_3 = SVC(kernel="rbf", random_state=42)
cv_scores_3 = cross_val_score(svm_3, X_train_3, y_train_3, cv=5)
print("\nCross Validation Score:", cv_scores_3)
print("Mean CV Accuracy      :", round((cv_scores_3.mean() * 100), 2), "\b%")
svm_3.fit(X_train_3, y_train_3)
y_pred_3 = svm_3.predict(X_test_3)

# =======================================
# Part e: Confusion Matrix For Each Model
# =======================================

print("\n========================")
print("Consfusion Matrix - PCA2")
print("========================\n")

cm_2 = confusion_matrix(y_test_2, y_pred_2)
print(cm_2)

print("\n========================")
print("Consfusion Matrix - PCA3")
print("========================\n")

cm_3 = confusion_matrix(y_test_3, y_pred_3)
print(cm_3)

# ===============================
# Part f: Accuracy For Each Model
# ===============================

print("\n===============")
print("Accuracy - PCA2")
print("===============\n")

accuracy = accuracy_score(y_test_2, y_pred_2)
print("Accuracy of PCA2 with SVM:", round((accuracy * 100), 2), "\b%")

print("\n===============")
print("Accuracy - PCA3")
print("===============\n")

accuracy = accuracy_score(y_test_3, y_pred_3)
print("Accuracy of PCA3 with SVM:", round((accuracy * 100), 2), "\b%")

# =========================================
# Part g: GridSearchCV for 3 PCA Components
# =========================================

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Create SVM model
svm_grid = SVC()

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_grid,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Train GridSearchCV using training data
grid_search.fit(X_train_3, y_train_3)

# Display best parameters
print("\n===================")
print("Grid Search Results")
print("===================\n")

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", round((grid_search.best_score_ * 100), 2), "\b%")
cm_grid = confusion_matrix(y_test_3, y_pred_3)

# ===================
# Test the Best Model
# ===================

best_svm = grid_search.best_estimator_
y_pred_grid = best_svm.predict(X_test_3)

# Confusion Matrix
cm_grid = confusion_matrix(y_test_3, y_pred_grid)
print("\n===========================")
print("Confusion Matrix - Best SVM")
print("===========================\n")
print(cm_grid)

# Accuracy
grid_accuracy = accuracy_score(y_test_3, y_pred_grid)
print("\nBest SVM Test Accuracy:", round((grid_accuracy * 100), 2), "\b%")