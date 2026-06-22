### Registration Number : 2023s20371
### Index Number        : s17618
### Course Code & Name  : IT 3005 - Data Mining
### Assignment 01       : Analysis of Brain Details Dataset

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load Data Set
file_path = r"A:\LMS\03. IT 3005\03. Datasets\01. Assignment 01\brain_details_EP.csv"
df = pd.read_csv(file_path)
df.info()

### 01. Boxplot of Gender vs Head Size
### ==================================

male = df[df['Gender'] == 1]['Head Size(cm^3)']
female = df[df['Gender'] == 2]['Head Size(cm^3)']

# Create boxplot
plt.figure(figsize=(8,5))
plt.boxplot([male, female], labels=['Male', 'Female'])

# Titles and labels
plt.title("Head Size by Gender")
plt.xlabel("Gender")
plt.ylabel("Head Size (cm³)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 02. Boxplot of Gender vs Brain Weight
### =====================================

male = df[df['Gender'] == 1]['Brain Weight(grams)']
female = df[df['Gender'] == 2]['Brain Weight(grams)']

# Create boxplot
plt.figure(figsize=(8,5))
plt.boxplot([male, female], labels=['Male', 'Female'])

# Titles and labels
plt.title("Brain Weight by Gender")
plt.xlabel("Gender")
plt.ylabel("Brain Weight (grams)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 03. Boxplot of Age Group vs Head Size
### =====================================

age_group_01 = df[df['Age Range'] == 1]['Head Size(cm^3)']
age_group_02 = df[df['Age Range'] == 2]['Head Size(cm^3)']

# Create a Boxplot
plt.figure(figsize=(8, 5))
plt.boxplot([age_group_01, age_group_02], labels=['Age Group 01', 'Age Group 02'])

# Title and Labels 
plt.title("Age Group vs Head Size")
plt.xlabel("Age Groups")
plt.ylabel("Head Size (cm³)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 04. Boxplot of Age Group vs Brain Weight
### =======================================

age_group_01 = df[df['Age Range'] == 1]['Brain Weight(grams)']
age_group_02 = df[df['Age Range'] == 2]['Brain Weight(grams)']

# Create a Boxplot
plt.figure(figsize=(8, 5))
plt.boxplot([age_group_01, age_group_02], labels=['Age Group 01', 'Age Group 02'])

# Title and Labels 
plt.title("Age Group vs Brain Weight")
plt.xlabel("Age Groups")
plt.ylabel("Brain Weight (grams)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 05. Scatter Plot of Head Size vs Brain Weight (by Gender)
### =========================================================

x = df['Head Size(cm^3)']
y = df['Brain Weight(grams)']

sns.scatterplot(
    x='Head Size(cm^3)',
    y='Brain Weight(grams)',
    hue=df['Gender'].map({1: 'Male', 2: 'Female'}),
    data=df
)

# Best fit line (linear regression)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='red')

plt.title("Head Size vs Brain Weight (by Gender)")
plt.xlabel("Head Size (cm³)")
plt.ylabel("Brain Weight (grams)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

### 06. Scatter Plot of Head Size vs Brain Weight (by Age Group)
### ============================================================

x = df['Head Size(cm^3)']
y = df['Brain Weight(grams)']

sns.scatterplot(
    x='Head Size(cm^3)',
    y='Brain Weight(grams)',
    hue=df['Age Range'].map({1:'Age Group 01', 2:'Age Group 02'}),
    data=df
)

# Best fit line (linear regression)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='red')

plt.title("Head Size vs Brain Weight (by Age Group)")
plt.xlabel("Head Size (cm³)")
plt.ylabel("Brain Weight (grams)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

### 07. Bar Chart of Mean Head Size by Gender
### =========================================

df.groupby(df['Gender'].map({1:'Male',2:'Female'}))['Head Size(cm^3)'].mean().plot(kind='bar')

plt.title("Average Head Size by Gender")
plt.xlabel("Gender")
plt.ylabel("Mean Head Size (cm³)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 08. Bar Chart of Mean Brain Weight by Gender
### ============================================

df.groupby(df['Gender'].map({1:'Male',2:'Female'}))['Brain Weight(grams)'].mean().plot(kind='bar')

plt.title("Average Brain Weight by Gender")
plt.xlabel("Gender")
plt.ylabel("Mean Brain Weight (grams)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 09. Bar Chart of Mean Head Size by Age Group
### ============================================

df.groupby(df['Age Range'].map({1:'Age Group 01',2:'Age Group 02'}))['Head Size(cm^3)'].mean().plot(kind='bar')

plt.title("Average Head Size by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Mean Head Size (cm³)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 10. Bar Chart of Mean Brain Weight by Age Group
### ===============================================

df.groupby(df['Age Range'].map({1:'Age Group 01',2:'Age Group 02'}))['Brain Weight(grams)'].mean().plot(kind='bar')

plt.title("Average Brain Weight by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Mean Brain Weight (grams)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 11. Histogram of Head Size
### ===========================

plt.hist(df['Head Size(cm^3)'], bins=20, edgecolor='black')

plt.title("Distribution of Head Size")
plt.xlabel("Head Size (cm³)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

### 12. Histogram of Brain Weight
### =============================

plt.hist(df['Brain Weight(grams)'], bins=20, edgecolor='black')

plt.title("Distribution of Brain Weight")
plt.xlabel("Brain Weight (grams)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()