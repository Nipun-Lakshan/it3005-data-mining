### Registration Number : 2023s20371
### Index Number        : s17618
### Course Code & Name  : IT 3005 - Data Mining
### Assignment 01       : Analysis of Covid Asia Details Dataset

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load Data Set
file_path = r"A:\LMS\03. IT 3005\03. Datasets\01. Assignment 01\covid_asia_details_EP.csv"
df = pd.read_csv(file_path)
df.info()

### 01. Top 10 Countries by Total Cases
### ===================================

top10_cases = df.nlargest(10, 'TotalCases')[['Country', 'TotalCases']]

plt.figure(figsize=(12, 6))
sns.barplot(data=top10_cases, x='Country', y='TotalCases')
plt.title('Top 10 Countries by Total COVID-19 Cases')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.6, linestyle='--')
plt.show()

### 02. Total Cases Per Million vs Total Deaths Per Million
### =======================================================

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='TotalCasesPerMillion', y='TotalDeathsPerMillion')
plt.title('Total Deaths per Million vs Total Cases per Million')
plt.xlabel('Total Cases per Million')
plt.ylabel('Total Deaths per Million')
plt.grid(linestyle='--', alpha=0.6)
plt.show()

### 03. Distribution of Total Cases Per Million
### ===========================================

plt.figure(figsize=(12, 6))
sns.histplot(df['TotalCasesPerMillion'], bins=20, kde=True)
plt.title('Distribution of Total Cases per Million Across Countries')
plt.xlabel('Total Cases per Million')
plt.ylabel('Number of Countries')
plt.grid(linestyle='--', alpha=0.6)
plt.show()