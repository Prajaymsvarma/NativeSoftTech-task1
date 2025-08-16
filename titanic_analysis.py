import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Display first 5 rows
print("Titanic Dataset Sample:")
print(titanic.head())

# Data Manipulation: Filtering passengers who survived
survived = titanic[titanic['survived'] == 1]

# Group by class and get average age and fare
grouped = survived.groupby('class').agg({'age': 'mean', 'fare': 'mean'})
print("\nAverage age and fare of survivors by class:")
print(grouped)

# Histogram of ages of passengers
plt.figure(figsize=(8,5))
plt.hist(titanic['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_histogram.png')
plt.show()

# Bar chart of survivors count by class
plt.figure(figsize=(6,4))
sns.countplot(x='class', hue='survived', data=titanic)
plt.title('Survivors by Passenger Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.savefig('survivors_by_class.png')
plt.show()

# Scatter plot: Age vs Fare colored by Survival
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic)
plt.title('Age vs Fare (Survivors highlight)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig('age_vs_fare_scatter.png')
plt.show()
