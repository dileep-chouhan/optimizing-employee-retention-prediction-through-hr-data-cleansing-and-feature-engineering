import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic HR data
num_employees = 500
data = {
    'EmployeeID': range(1, num_employees + 1),
    'Age': np.random.randint(20, 60, size=num_employees),
    'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], size=num_employees),
    'YearsOfExperience': np.random.randint(0, 30, size=num_employees),
    'Salary': np.random.randint(40000, 150000, size=num_employees),
    'JobSatisfaction': np.random.randint(1, 6, size=num_employees), # 1-5 scale
    'PerformanceRating': np.random.randint(1, 6, size=num_employees), # 1-5 scale
    'LeftCompany': np.random.choice([0, 1], size=num_employees, p=[0.8, 0.2]), # 0: stayed, 1: left
    'Bonus': np.random.choice([0,1], size=num_employees, p=[0.7,0.3]) # 0: no bonus, 1: bonus
}
df = pd.DataFrame(data)
# Introduce some messy data: missing values and inconsistencies
df.loc[np.random.choice(df.index, size=50), 'Salary'] = np.nan # Introduce missing salaries
df.loc[np.random.choice(df.index, size=20), 'Department'] = 'Unknown' # Introduce unknown departments
df['YearsOfExperience'] = df['YearsOfExperience'].astype(str).str.replace('10','10.0') # Introduce inconsistent data type
# --- 2. Data Cleaning ---
# Handle missing values
df['Salary'].fillna(df['Salary'].mean(), inplace=True) # Impute missing salaries with the mean
# Handle inconsistent data types
df['YearsOfExperience'] = pd.to_numeric(df['YearsOfExperience'], errors='coerce')
df['YearsOfExperience'].fillna(df['YearsOfExperience'].mean(), inplace=True) # Impute missing values after type conversion
# Handle inconsistent categories
df['Department'].replace('Unknown', 'Other', inplace=True)
# --- 3. Feature Engineering ---
# Create a new feature: TotalCompensation
df['TotalCompensation'] = df['Salary'] + (df['Bonus'] * df['Salary'] * 0.1) # Assuming 10% bonus
# --- 4. Analysis ---
# Analyze employee turnover by department
turnover_by_department = df.groupby('Department')['LeftCompany'].mean()
print("Employee Turnover by Department:")
print(turnover_by_department)
# Analyze the relationship between job satisfaction and turnover
correlation = df['JobSatisfaction'].corr(df['LeftCompany'])
print(f"\nCorrelation between Job Satisfaction and Turnover: {correlation}")
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(x=turnover_by_department.index, y=turnover_by_department.values)
plt.title('Employee Turnover Rate by Department')
plt.xlabel('Department')
plt.ylabel('Turnover Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save the plot to a file
output_filename = 'turnover_by_department.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
sns.scatterplot(x='JobSatisfaction', y='LeftCompany', data=df, hue='Department')
plt.title('Job Satisfaction vs. Turnover')
plt.xlabel('Job Satisfaction')
plt.ylabel('Left Company (0=No, 1=Yes)')
plt.tight_layout()
output_filename2 = 'job_satisfaction_vs_turnover.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")