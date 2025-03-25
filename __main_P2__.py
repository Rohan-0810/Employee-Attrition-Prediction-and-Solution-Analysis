#  ____  _   _    _    ____  _____       ____  
# |  _ \| | | |  / \  / ___|| ____|     |___ \ 
# | |_) | |_| | / _ \ \___ \|  _|         __) |
# |  __/|  _  |/ ___ \ ___) | |___       / __/ 
# |_|   |_| |_/_/   \_\____/|_____|     |_____|

# Importing Libraries
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import seaborn as sns

# Importing the data set
df = pd.read_excel("/Users/_.rohan._/Desktop/EMPLOYEE_ATTRITION/EXTRA datasets/EA with names.xlsx")

# List of columns to plot
columns_to_plot = [
    'DistanceFromHome', 'Education', 'RelationshipSatisfaction', 
    'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
    'JobSatisfaction', 'PercentSalaryHike', 
    'TrainingTimesLastYear', 'Age', 'MonthlyIncome', 'YearsAtCompany',
    'TotalWorkingYears', 'Department', 'BusinessTravel', 'PerformanceRating'
]

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Loop through each column to create histograms
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 4, i)  # Create a 4x4 grid of subplots
    sns.histplot(data=df, x=column, hue='Attrition', multiple='stack', bins=30, kde=True)
    plt.title(f'{column} by Attrition')
    plt.xlabel(column)
    plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# Main Algorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample

# Load the dataset
file_path = "/mnt/data/EA with names.xlsx"
df=pd.read_excel("/Users/_.rohan._/Desktop/EMPLOYEE_ATTRITION/EXTRA datasets/EA with names.xlsx")
# Drop irrelevant columns
df_cleaned = df.drop(columns=["Name", "EmployeeCount", "Over18", "StandardHours"]) 

# Encode categorical variables
categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns
label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col, encoder in label_encoders.items():
    df_cleaned[col] = encoder.fit_transform(df_cleaned[col])

# Separate majority and minority classes
df_majority = df_cleaned[df_cleaned["Attrition"] == 0]
df_minority = df_cleaned[df_cleaned["Attrition"] == 1]

# Balance dataset using undersampling
df_majority_downsampled = resample(df_majority, replace=False,n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# Separate features and target
X_balanced = df_balanced.drop(columns=["Attrition"])
y_balanced = df_balanced["Attrition"]

# Standardize numerical features
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)

# Split data into training and test sets
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_balanced_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Train models
models_balanced = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42)
}

# Evaluate models
results_balanced = {}
for name, model in models_balanced.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred_bal = model.predict(X_test_bal)
    accuracy = accuracy_score(y_test_bal, y_pred_bal)
    f1 = f1_score(y_test_bal, y_pred_bal)
    results_balanced[name] = {"Accuracy": accuracy, "F1 Score": f1}

# Convert results to DataFrame
results_balanced_df = pd.DataFrame(results_balanced).T
print("Model Performance:\n", results_balanced_df)

# Select best model (SVM) and predict on full dataset
best_model_balanced = models_balanced["Support Vector Machine"]
X_full_scaled = scaler.transform(df_cleaned.drop(columns=["Attrition"]))
y_full_pred = best_model_balanced.predict(X_full_scaled)

# Add predictions to original dataset
df["Predicted_Attrition"] = y_full_pred

# Display sample results
print("\nPredicted vs Actual Attrition:\n", df[["Name", "Attrition", "Predicted_Attrition"]])

########################
# Feature Importance Analysis
########################

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance

# Drop irrelevant columns
df_cleaned = df.drop(columns=["Name", "EmployeeCount", "Over18", "StandardHours","NumCompaniesWorked","Predicted_Attrition","Department","Gender","MaritalStatus"])

# Encode categorical variables
categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns
label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col, encoder in label_encoders.items():
    df_cleaned[col] = encoder.fit_transform(df_cleaned[col])

# Separate features and target variable
X = df_cleaned.drop(columns=["Attrition"])
y = df_cleaned["Attrition"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42)
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"Accuracy": accuracy, "F1 Score": f1}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("Model Performance:\n", results_df)

# Select best models for feature importance analysis
logreg_model = models["Logistic Regression"]
rf_model = models["Random Forest"]

# Logistic Regression - Feature Importance (Coefficients)
feature_importance_lr = pd.DataFrame({
    "Feature": X.columns,
    "Importance": logreg_model.coef_[0]
}).sort_values(by="Importance", ascending=False)

# Random Forest - Feature Importance
feature_importance_rf = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Get top 10 features for both models
top_features_lr = feature_importance_lr.head(10)
top_features_rf = feature_importance_rf.head(10)

# Visualization of Top 10 Important Features for Logistic Regression
plt.figure(figsize=(10, 5))
plt.barh(top_features_lr["Feature"], top_features_lr["Importance"], color="blue")
plt.xlabel("Coefficient Value (Importance)")
plt.ylabel("Feature")
plt.title("Top 10 Features Affecting Attrition - Logistic Regression")
plt.gca().invert_yaxis()
plt.show()

# Visualization of Top 10 Important Features for Random Forest
plt.figure(figsize=(10, 5))
plt.barh(top_features_rf["Feature"], top_features_rf["Importance"], color="green")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Features Affecting Attrition - Random Forest")
plt.gca().invert_yaxis()
plt.show()

# Permutation Importance (Using Logistic Regression)
perm_importance = permutation_importance(logreg_model, X_test, y_test, scoring="accuracy")
perm_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# Permutation Importance (Using Random Forest)
perm_importance_rf = permutation_importance(rf_model, X_test, y_test, scoring="accuracy")
perm_importance_rf_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance_rf.importances_mean
}).sort_values(by="Importance", ascending=False)

# Display top 5 most important features
print("\nTop 10 Features from Permutation Importance:\n", perm_importance_df.head(5))
print("\nTop 10 Features from Permutation Importance:\n", perm_importance_rf_df.head(5))

#  _____ _____ ____  __  __ ___ _   _    _  _____ _____ ____     _   _  ___   _    _
# |_   _| ____|  _ \|  \/  |_ _| \ | |  / \|_   _| ____|  _ \   | \ | |/ _ \ | |  | |
#   | | |  _| | |_) | |\/| || ||  \| | / _ \ | | |  _| | | | |  |  \| | | | || |  | |
#   | | | |___|  _ <| |  | || || |\  |/ ___ \| | | |___| |_| |  | |\  | |_| || |/\| | 
#   |_| |_____|_| \_\_|  |_|___|_| \_/_/   \_\_| |_____|____/   |_| \_|\___/ |__/\__|