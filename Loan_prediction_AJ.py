#Importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

#Loading the data set 
df = pd.read_csv(r"C:\Users\abhis\Downloads\loan_detection.csv")
print(df.head())
print(df.shape)

#EDA 
#Finding missing or duplicate values 
print(df.info())
print(df.describe())
print(df.nunique())
print(df.isnull().sum())
print(df.duplicated().sum())

print(df.columns)
print(df['Loan_Status_label'])
print(df['Loan_Status_label'].value_counts())

# Imbalance Data ratio and reprensentation
print(round(len(df[df['Loan_Status_label']==0])/len(df)*100, 2))
print(round(len(df[df['Loan_Status_label']==1])/len(df)*100, 2))

# # Analyze categorical variables
# categorical_columns = [col for col in df.columns if 'job_' in col or 'education_' in col or 'marital_' in col]
# for col in categorical_columns:
#     sns.countplot(x=col, data=df)
#     plt.title(f'Distribution of {col}')
#     plt.xticks(rotation=90)
#     plt.show()

plt.pie(df['Loan_Status_label'].value_counts(), autopct='%1.0f%%',labels=['Not Eligible','Eligible'], 
            startangle=60,shadow=True,explode=[0,0.2])
plt.title('Imbalanced Data Visulaization')
plt.show()

#Outlier or Anomalies
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1  # Interquartile Range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]    
    return df

numerical_columns = ['age', 'campaign', 'pdays', 'previous']  

df_cleaned = remove_outliers_iqr(df, numerical_columns)

print("Original dataset shape:", df.shape)
print("Cleaned dataset shape (after removing outliers):", df_cleaned.shape)

# Pairplot to analyze relationships after cleaning
sns.pairplot(df_cleaned[['age', 'campaign', 'previous', 'Loan_Status_label']], hue='Loan_Status_label')
plt.title('Pairplot after cleaning')
plt.show()

# Visualizing distribution of age and other numeric columns
numeric_columns = ['age', 'campaign', 'pdays', 'previous']
for col in numeric_columns:
    sns.histplot(df_cleaned[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

#Split the dataset into features and target variable
X = df_cleaned.drop(columns=['Loan_Status_label'])  # Features
y = df_cleaned['Loan_Status_label']  # Target (Loan Approved or Not)

#Check class distribution before resampling
print("Class distribution before SMOTE:", Counter(y))

#Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Check class distribution after resampling
print("Class distribution after SMOTE:", Counter(y_resampled))

#Splitting your data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

#Feature scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model Building
#Initialize Logistic Regression, Random Forest, and Decision Tree models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

# Train Logistic Regression
log_reg.fit(X_train_scaled, y_train)

# Train Random Forest
rf_model.fit(X_train, y_train)  

# Train Decision Tree
dt_model.fit(X_train, y_train)  

# Model Evaluation
# Logistic Regression Evaluation
y_pred_log_reg = log_reg.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression Accuracy: {log_reg_accuracy:.2f}')

# Random Forest Evaluation
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Decision Tree Evaluation
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')

# Confusion Matrix and Classification Report for Random Forest
print("\nConfusion Matrix for Random Forest:")
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.show()

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Hyperparameter Tuning using GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model from GridSearchCV
best_rf_model = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')

# Evaluate the tuned model
y_pred_best_rf = best_rf_model.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, y_pred_best_rf)
print(f'Best Random Forest Accuracy (after tuning): {best_rf_accuracy:.2f}')

# Confusion Matrix and Heatmap (After Tuning)
print("\nConfusion Matrix for Random Forest (After Tuning):")
conf_matrix_after = confusion_matrix(y_test, y_pred_best_rf)
sns.heatmap(conf_matrix_after, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (After Tuning)')
plt.show()

#Classification Report (Optional for After Tuning)
print("\nClassification Report for Random Forest (After Tuning):")
print(classification_report(y_test, y_pred_best_rf))
