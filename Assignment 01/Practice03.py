## Registration Number : 2023s20371
### Index Number        : s17618
### Course Code & Name  : IT 3005 - Data Mining
### Assignment 01       : Analysis of Student Details Dataset

## Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load Data Set
file_path = r"A:\LMS\03. IT 3005\03. Datasets\01. Assignment 01\Data Vis_student details.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")
df.info()

### 01. Gender Distribution
### =======================

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle='--')
plt.show()

### 02. Height Distribution
### =======================

plt.figure(figsize=(8,5))
sns.histplot(df["Your height in cm"].dropna(), bins=10, kde=True, color="skyblue")
plt.title("Height Distribution")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.xticks([])
plt.grid(alpha=0.6, linestyle='--')
plt.show()

### 03. Height by Gender
### ====================

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Gender", y="Your height in cm")
plt.title("Height Comparison by Gender")
plt.xlabel("Gender")
plt.ylabel("Height (cm)")
plt.yticks([])
plt.show()