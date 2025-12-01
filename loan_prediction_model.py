# IMPORT LIBRARY
import os
import json
import joblib
import warnings
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline

# LOAD DATA
loan = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)

# DATA UNDERSTANDING
print("Amount of rows and columns:", loan.shape)
display("The first 5 rows:", loan.head())
display("The last 5 rows:", loan.tail())
print("\nAll of the name of columns:\n", loan.columns)
print("\nSummary of the DataFrame:\n", loan.info())
print("\nSummary statistics for numerical columns:\n", loan.describe())

numerical_cols = loan.select_dtypes(include=np.number).columns.tolist()
categorical_cols = loan.select_dtypes(include='object').columns.tolist()

print("Numerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# FEATURE ENGINEERING
# Define a list of columns to drop
irrelevant_columns = [
    # unique id
    'Unnamed: 0', 'id', 'member_id',
    # free text
    'url', 'desc',
    #  all null/constant/other
    'zip_code', 'annual_inc_joint', ' dti_joint', 'verification_status_joint',
    'open_acc_6m', 'open_il', 'open_il_6m', 'open_il_12m', 'open_il_24m',
    'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open',
    'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
    'inq_fi', 'total_cu_tl', 'inq_last_12m', 'mths_since_last_record',
    'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
    # expert judgment
     'sub_grade']

# Drop the irrelevant columns if they exist in the DataFrame
columns_to_drop = [col for col in irrelevant_columns if col in loan.columns]
if columns_to_drop:
    loan.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped irrelevant columns: {columns_to_drop}")
else:
    print("No specified irrelevant columns found to drop.")

print(f"Shape of loan_cleaned after dropping columns: {loan.shape}")

# Checking Target Variable
print("Loan status distribution (percentage):")
display(loan['loan_status'].value_counts(normalize=True) * 100)

bad_status = ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']
loan['bad_status'] = np.where(loan['loan_status'].isin(bad_status).astype(int), 1, 0)
percentage_bad_status = loan['bad_status'].mean() * 100
print(f"\nPercentage of bad status: {percentage_bad_status:.2f}%")
loan.drop('loan_status', axis=1, inplace=True)

# Data Transformation
# 'emp_length' Column
loan['emp_length'].unique()
loan['emp_length_num'] = loan['emp_length'].str.replace(r'\D', '', regex=True)
print(loan['emp_length_num'].describe())
loan.drop('emp_length', axis=1, inplace=True)

# 'term' Column
loan['term']
loan['term_num'] = loan['term'].str.replace('months','').astype(float)
print(loan['term_num'].describe())
loan.drop('term', axis=1, inplace=True)

# 'earliest_cr_line' column
loan['earliest_cr_line'].unique()
# The earliest_cr_line column contains strings, such as "Jan-15"
loan['earliest_cr_line_bY'] = pd.to_datetime(loan['earliest_cr_line'], format='%b-%y')
latest_date = loan['earliest_cr_line_bY'].max()
earliest_date = loan['earliest_cr_line_bY'].min()
print("The latest date:", latest_date)
print("The earliest date", earliest_date)

reference_year = pd.Timestamp('2017-12-01')
loan.loc[loan['earliest_cr_line_bY'].dt.year > reference_year.year, 'earliest_cr_line_bY'] -= pd.offsets.DateOffset(years=100)
# Correct column name 'earliest_cr_line' and format '%b-%y'
loan['months_since_earliest_cr_line'] = round(reference_year - loan['earliest_cr_line_bY']).dt.days / 30
print(loan['months_since_earliest_cr_line'].describe())
loan.drop(['earliest_cr_line', 'earliest_cr_line_bY'], axis=1, inplace=True)

# ' issue_d' column
loan['issue_d']
# The issue_d column contains strings, such as "Jan-15"
loan['issue_d_bY'] = pd.to_datetime(loan['issue_d'], format='%b-%y')
latest_date = loan['issue_d_bY'].max()
earliest_date = loan['issue_d_bY'].min()
print("The latest date:", latest_date)
print("The earliest date:", earliest_date)

reference_year = pd.Timestamp('2017-12-01')
loan.loc[loan['issue_d_bY'].dt.year > reference_year.year, 'issue_d_bY'] -= pd.offsets.DateOffset(years=100)
# Correct column name 'issue_d' and format '%b-%y'
loan['months_since_issue_d'] = round(reference_year - loan['issue_d_bY']).dt.days / 30
print(loan['months_since_issue_d'].describe())
loan.drop(['issue_d', 'issue_d_bY'], axis=1, inplace=True)

# 'last_pymnt_d' column
loan['last_pymnt_d']
# The last_pymnt_d column contains strings, such as "Jan-15"
loan['last_pymnt_d_bY'] = pd.to_datetime(loan['last_pymnt_d'], format='%b-%y')
latest_date = loan['last_pymnt_d_bY'].max()
earliest_date = loan['last_pymnt_d_bY'].min()
print("The latest date:", latest_date)
print("The earliest date:", earliest_date)

# Correct column name 'last_pymnt_d' and format '%b-%y'
loan['months_since_last_pymnt_d'] = round(reference_year - loan['last_pymnt_d_bY']).dt.days / 30
print(loan['months_since_last_pymnt_d'].describe())
loan.drop(['last_pymnt_d', 'last_pymnt_d_bY'], axis=1, inplace=True)

# 'next_pymnt_d' column
loan['next_pymnt_d']
# The next_pymnt_d column contains strings, such as "Jan-15"
loan['next_pymnt_d_bY'] = pd.to_datetime(loan['next_pymnt_d'], format='%b-%y')
latest_date = loan['next_pymnt_d_bY'].max()
earliest_date = loan['next_pymnt_d_bY'].min()
print("The latest date:", latest_date)
print("The earliest date:", earliest_date)

# Correct column name 'next_pymnt_d' and format '%b-%y'
loan['months_since_next_pymnt_d'] = round(reference_year - loan['next_pymnt_d_bY']).dt.days / 30
print(loan['months_since_next_pymnt_d'].describe())
loan.drop(['next_pymnt_d', 'next_pymnt_d_bY'], axis=1, inplace=True)

# 'last_credit_pull_d' column
loan['last_credit_pull_d']
# The last_credit_pull_d column contains strings, such as "Jan-15"
loan['last_credit_pull_d_bY'] = pd.to_datetime(loan['last_credit_pull_d'], format='%b-%y')
latest_date = loan['last_credit_pull_d_bY'].max()
earliest_date = loan['last_credit_pull_d_bY'].min()
print("The latest date:", latest_date)
print("The earliest date:", earliest_date)

# Correct column name 'last_credit_pull_d' and format '%b-%y'
loan['months_since_last_credit_pull_d'] = round(reference_year - loan['last_credit_pull_d_bY']).dt.days / 30
print(loan['months_since_last_credit_pull_d'].describe())
loan.drop(['last_credit_pull_d','last_credit_pull_d_bY'], axis=1, inplace=True)

# EXPLORATORY DATA ANALYSIS
#  Check Cardinality Data
print(loan.select_dtypes(include='object').nunique())
loan.drop(['emp_title', 'title'], axis=1, inplace=True)

print(loan.select_dtypes(exclude='object').nunique())
loan.drop(['policy_code','dti_joint'], axis=1, inplace=True)

for col in loan.select_dtypes(include='object').columns.tolist():
    print(f"Distribution of unique values in column '{col}':")
    print(loan[col].value_counts(normalize=True, dropna=False))
    print("\n")
loan.drop(['pymnt_plan', 'application_type'], axis=1, inplace=True)

# Univariate Analysis
# Categorical Column
categorical_var = loan.select_dtypes(include='object').columns
print(categorical_cols)

plt.figure(figsize=(20, 6))
for i, col in enumerate(categorical_var,1):
    plt.subplot(2, 3, i)
    sns.countplot(data=loan, x=col, color='purple')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=90)
    plt.tight_layout()
plt.show()

plt.style.use('ggplot')

for column in categorical_var:
    plt.figure(figsize=(20, 4))
    plt.subplot(121)
    loan[column].value_counts().plot(kind='bar', color='teal')
    plt.xlabel(column)
    plt.ylabel('Number of Customers')
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()

# Numerical Column
numerical_var = loan.select_dtypes(exclude='object').columns.tolist()
print(numerical_var)

plt.style.use('ggplot')

average = loan[numerical_var].mean()
std = loan[numerical_var].std()
median = loan[numerical_var].median()
mode = loan[numerical_var].mode().iloc[0]

for column in numerical_var:
    plt.figure(figsize=(20, 4))
    plt.subplot(121)
    sns.histplot(loan[column], kde=True, color='teal')
    plt.axvline(average[column], color='red', linestyle='solid', linewidth=3, label='Average')
    plt.axvline(median[column], color='green', linestyle='dashed', linewidth=3, label='Median')
    plt.axvline(mode[column], color='blue', linestyle='dotted', linewidth=3, label='Mode')
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()

# Bivariate Analysis
plt.style.use('ggplot')

for column in categorical_var:
    plt.figure(figsize=(20, 4))
    plt.subplot(121)
    sns.countplot(x=loan[column], hue=loan['bad_status'], palette='Set2')
    plt.title(f'Distribution of {column} by Bad Status')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Checking and Handling Missing Value
missing_percent = loan.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

print("Missing Values Summary (columns with > 0% missing):")
print(missing_percent)
print("\n")

missing_threshold = 70
cols_to_drop_high_missing = missing_percent[missing_percent > missing_threshold].index.tolist()

if cols_to_drop_high_missing:
  print(f"Dropping columns with > {missing_threshold}% missing values: {cols_to_drop_high_missing}")
  loan.drop(columns=cols_to_drop_high_missing, axis = 1, inplace=True)
  print(f"Shape after dropping high missing columns: {loan.shape}")

# Data Imputation
# Define skew_threshold. This variable was used but not defined in the original snippet.
skew_threshold = 0.5 # A common threshold, adjust as needed based on data distribution

# Take column missing value after dropped
missing_percent_current = loan.isnull().mean() * 100
cols_to_impute = missing_percent_current[missing_percent_current > 0].index.tolist()
print(cols_to_impute)

for col in cols_to_impute:
    # Impute numeric columns with mean or median based on skew
    if pd.api.types.is_numeric_dtype(loan[col]):
        # Calculate skew only if there are enough non-null values in the *original* column
        if loan[col].count() > 1:
            skew_val = loan[col].skew()
        else:
            skew_val = 0 # Cannot calculate skew if too few values

        if abs(skew_val) < skew_threshold:
            fill_val = loan[col].mean()
            method = "mean"
        else:
            fill_val = loan[col].median()
            method = "median"
        loan[col].fillna(fill_val, inplace=True)
        print(f"Imputed '{col}' with {method}: {fill_val}")

#  Label Encoding
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

categorical_var = loan.select_dtypes(include='object').columns
print(categorical_cols)

for col in categorical_var:
    loan[col] = label.fit_transform(loan[col])

loan.head()

# Create New Feature
# 1. Create loan_to_income_ratio
# Add a small epsilon to annual_inc to avoid division by zero
epsilon = 1e-6
loan['loan_to_income_ratio'] = loan['loan_amnt'] / (loan['annual_inc'] + epsilon)
print(f"'loan_to_income_ratio' created. First 5 values:\n{loan['loan_to_income_ratio'].head()}")

# 2. Create installment_to_income_ratio
loan['installment_to_income_ratio'] = loan['installment'] / (loan['annual_inc'] + epsilon)
print(f"'installment_to_income_ratio' created. First 5 values:\n{loan['installment_to_income_ratio'].head()}")

# 3. Initialize a small constant epsilon to prevent division by zero
epsilon = 1e-6

# 4. Create revolving_burden_index
loan['revolving_burden_index'] = loan['revol_bal'] / (loan['annual_inc'] + epsilon)
print(f"'revolving_burden_index' created. First 5 values:\n{loan['revolving_burden_index'].head()}")

# 5. Create principal_repayment_ratio
loan['principal_repayment_ratio'] = loan['total_rec_prncp'] / (loan['loan_amnt'] + epsilon)
print(f"'principal_repayment_ratio' created. First 5 values:\n{loan['principal_repayment_ratio'].head()}")

# 6. Create payment_progress_ratio
loan['payment_progress_ratio'] = loan['total_pymnt'] / (loan['loan_amnt'] + epsilon)
print(f"'payment_progress_ratio' created. First 5 values:\n{loan['payment_progress_ratio'].head()}")

# 7. Create interest_payment_indicator
loan['interest_payment_indicator'] = loan['total_rec_int'] / (loan['total_pymnt'] + epsilon)
print(f"'interest_payment_indicator' created. First 5 values:\n{loan['interest_payment_indicator'].head()}")

# 8. Create funded_ratio
loan['funded_ratio'] = loan['funded_amnt'] / (loan['loan_amnt'] + epsilon)
print(f"'funded_ratio' created. First 5 values:\n{loan['funded_ratio'].head()}")

# 9-11. Interaction Terms
loan['loan_int_product'] = loan['loan_amnt'] * loan['int_rate']
loan['dti_inc_product'] = loan['dti'] * loan['annual_inc']
loan['revol_bal_util'] = loan['revol_bal'] * loan['revol_util']

# 12-14. Polynomial Features
loan['annual_inc_sq'] = loan['annual_inc'] ** 2
loan['dti_sq'] = loan['dti'] ** 2

loan['collections_per_acc'] = np.where(
    (loan['total_acc'].notna()) & (loan['total_acc'] != 0),
    loan['collections_12_mths_ex_med'] / loan['total_acc'],
    np.nan
)

# Replace inf values with NaN (if any)
loan.replace([np.inf, -np.inf], np.nan, inplace=True)

print("New features added to DataFrame. Displaying head with new columns:")
loan.head()

# Time-Based Split
# Sort by months_since_issue_d (smaller values = more recent issues)
loan_sorted = loan.sort_values(by='months_since_issue_d', ascending=True).reset_index(drop=True)

split_point = int(len(loan_sorted) * 0.8)

X_train_time = loan_sorted.iloc[:split_point].drop('bad_status', axis=1)
y_train_time = loan_sorted.iloc[:split_point]['bad_status']

X_test_time = loan_sorted.iloc[split_point:].drop('bad_status', axis=1) # This will be the basis for final X_test
y_test_time = loan_sorted.iloc[split_point:]['bad_status'] # This will be the final y_test

print(f"Shape of X_train_time (after initial time split): {X_train_time.shape}")
print(f"Shape of y_train_time (after initial time split): {y_train_time.shape}")
print(f"Shape of X_test_time (raw): {X_test_time.shape}")
print(f"Shape of y_test_time (raw): {y_test_time.shape}")

# Handling Imbalance Data
# Apply SMOTE to the initial training data
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_smoted, y_train_smoted = sm.fit_resample(X_train_time, y_train_time)

print(f"\nShape of X_train_smoted (after SMOTE): {X_train_smoted.shape}")
print(f"Shape of y_train_smoted (after SMOTE): {y_train_smoted.shape}")
print("Target distribution in y_train_smoted (after SMOTE):\n", y_train_smoted.value_counts(normalize=True) * 100)

# Standardization
# Scale the SMOTEd training data and the time-based test data
scaler = StandardScaler()

# Fit ONLY on the SMOTE-resampled training set (X_train_smoted) and transform
# Convert scaled arrays back to DataFrame to preserve column names
X_train_scaled = scaler.fit_transform(X_train_smoted)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_smoted.columns)

# Transform the time-based test set (X_test_time) using training statistics
X_test_scaled = scaler.transform(X_test_time)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_time.columns)

print("\nData scaled.")
print(f"Shape of X_train_scaled (SMOTEd and scaled): {X_train_scaled.shape}")
print(f"Shape of X_test_scaled (time-based test, scaled): {X_test_scaled.shape}")

# Train Test Split (Train and Validation Set)
# Split the SMOTEd and Scaled training data into actual training and validation sets
X_train_actual, X_val, y_train_actual, y_val = train_test_split(
    X_train_scaled, y_train_smoted, test_size=0.2, stratify=y_train_smoted, random_state=42
)

print("\n--- Final Train/Validation/Test Sets ---")
print(f"Shape of X_train_actual: {X_train_actual.shape}")
print(f"Shape of y_train_actual: {y_train_actual.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_val: {y_val.shape}")
print(f"Shape of X_test (final, scaled): {X_test_scaled.shape}")
print(f"Shape of y_test (final, raw): {y_test_time.shape}")

print("\nClass distribution in y_train_actual (percentage):")
print(y_train_actual.value_counts(normalize=True) * 100)

print("\nClass distribution in y_val (percentage):")
print(y_val.value_counts(normalize=True) * 100)

# Re-assign global variables for consistency with existing notebook cells
# X_train and y_train here represent the SMOTEd and scaled training data (before actual/val split)
# as used by some existing cells (e.g., initial model training, RandomizedSearchCV for RF).
# X_test and y_test are the final scaled test set from the time-based split.
X_train = X_train_scaled
y_train = y_train_smoted
X_test = X_test_scaled
y_test = y_test_time

# Also ensure X_train_time and X_test_time (unscaled) are available if needed for PSI
# (as used in cell 8MPdcMNE5pNv and the PSI monitoring function)
X_train_time = X_train_time # Retain the unscaled time-based training set
X_test_time = X_test_time # Retain the unscaled time-based test set

# MODEL
# Initialize XGBoost
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    verbosity=0,
    random_state=42
)

# Modify the xgb_clf.fit call to use X_train_actual and eval_set with X_val
try:
    xgb_clf.fit(
        X_train_actual, y_train_actual,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
except TypeError:
    print("Warning: early_stopping_rounds not supported, fitting without it.")
    xgb_clf.fit(X_train_actual, y_train_actual)

# Logistic Regression (already trained on X_train, which is X_train_actual + X_val's combined data, so re-train on X_train_actual if consistency is paramount)
log_clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
log_clf.fit(X_train_actual, y_train_actual)

# Random Forest with RandomizedSearchCV (re-run as X_train_actual is smaller than X_train)
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=4,
    scoring='roc_auc',
    cv=2,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train_actual, y_train_actual) # Fit on X_train_actual
rf_clf = search.best_estimator_
print("RF best params:", search.best_params_)

# Calibration : calibrate RF probabilities (isotonic)
calibrated_rf = CalibratedClassifierCV(rf_clf, cv=2, method='isotonic')
calibrated_rf.fit(X_train_actual, y_train_actual) # Fit on X_train_actual

# Update models dictionary with newly trained models
models = {
    "Logistic": log_clf,
    "RandomForest": calibrated_rf,
    "XGBoost": xgb_clf
}

# Evaluate Models
eval_results = {}
for name, m in models.items():
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(X_test)[:, 1]
    else:
        proba_raw = m.decision_function(X_test)
        proba_raw = np.clip(proba_raw, -10, 10) # Clip for numerical stability
        proba = 1 / (1 + np.exp(-proba_raw))
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    eval_results[name] = {
        "auc": float(auc),
        "brier": float(brier),
        "confusion": cm.tolist(),
        "report": report
    }
    print(f"\n{name} -> AUC: {auc:.4f} | Brier: {brier:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_test, preds))
    
# Final Evaluation
OUTDIR = 'output'
os.makedirs(OUTDIR, exist_ok=True)

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Save evaluation results
save_json(eval_results, os.path.join(OUTDIR, "eval_results.json"))

# Select best model by AUC
best_name = max(eval_results, key=lambda k: eval_results[k]['auc'])
print("\nBest model by AUC:", best_name)
best_model = models[best_name]

# Define feature_names for SHAP (all columns in X_train are features)
# Correctly get feature names from the original DataFrame before scaling
feature_names = X_train_time.columns.tolist()

# Explainability (SHAP) - only for tree models (RF/XGB)
try:
    import shap
    import matplotlib.pyplot as plt # Ensure matplotlib is imported for plotting
    print("\nComputing SHAP (limited to 500 samples for speed). Only top 15 features are displayed by default for readability...")

    tree_model_for_shap = None # Initialize to None

    # Explicitly use the uncalibrated base estimator for SHAP if available and appropriate
    if best_name == "RandomForest":
        # rf_clf is the uncalibrated RandomForestClassifier from RandomizedSearchCV
        tree_model_for_shap = rf_clf
    elif best_name == "XGBoost":
        # xgb_clf is the uncalibrated XGBClassifier directly trained
        tree_model_for_shap = xgb_clf
    elif best_name == "XGBoost_Tuned":
        # xgb_tuned_clf is the tuned XGBClassifier from RandomizedSearchCV
        tree_model_for_shap = xgb_tuned_clf


    if tree_model_for_shap is not None and isinstance(tree_model_for_shap, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(tree_model_for_shap)
        max_samples = 500
        # Use X_test if available and large enough, otherwise X_train
        X_shap = X_test if X_test.shape[0] <= max_samples else X_test[:max_samples]
        shap_values = explainer.shap_values(X_shap)
        if feature_names and len(feature_names) == X_shap.shape[1]:
            shap.summary_plot(shap_values, X_shap, feature_names=feature_names, max_display=15) # max_display=15 limits the shown features
        else:
            shap.summary_plot(shap_values, X_shap, max_display=15) # max_display=15 limits the shown features
        plt.show() # Display the SHAP plot
    else:
        print(f"SHAP TreeExplainer skipped: best model '{best_name}' is not a directly supported tree-based model for SHAP TreeExplainer, or its base estimator could not be extracted.")

except Exception as e:
    print("SHAP error or not available:", e)

# PSI monitoring function
def psi(expected, actual, buckets=10):
    expected = np.array(expected).astype(float)
    actual = np.array(actual).astype(float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)
    exp_perc = exp_counts / exp_counts.sum()
    act_perc = act_counts / act_counts.sum()
    exp_perc = np.where(exp_perc == 0, 1e-6, exp_perc)
    act_perc = np.where(act_perc == 0, 1e-6, act_perc)
    psi_val = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
    return float(psi_val)

# Use all features in X_train_time for PSI calculation
psi_features = X_train_time.columns.tolist() # Calculate PSI for ALL features
psi_report_vals = {}
for f in psi_features:
    # Use X_train_time and X_test_time directly as they contain the processed data
    if f in X_train_time.columns and f in X_test_time.columns:
        psi_report_vals[f] = psi(X_train_time[f].dropna().values, X_test_time[f].dropna().values, buckets=10)
    else:
        psi_report_vals[f] = None

print("\nPSI values for ALL features:")
print(psi_report_vals)
save_json(psi_report_vals, os.path.join(OUTDIR, "psi_values.json"))

# Tentukan path psi_alert.py (pastikan file ini ada di folder OUTDIR atau sesuaikan path)
psi_alert_path = os.path.join(OUTDIR, "psi_alert.py")

# Save model artifacts, model_card, FastAPI template
# The 'scaler' object is defined and fit on numerical columns of X_train.
# Since all columns in X_train are now numerical and scaled, the 'scaler' can act as the preprocessor.
preprocessor = scaler
pipeline_best = make_pipeline(preprocessor, best_model)
best_model_path = os.path.join(OUTDIR, f"best_model_{best_name}.joblib")
joblib.dump(pipeline_best, best_model_path)
print("Saved best model pipeline to", best_model_path)

model_card = {
    "model_name": best_name,
    "date": datetime.now().isoformat(),
    "features": feature_names,
    "metrics": eval_results.get(best_name, {}),
    "notes": "Trained with cleaned LendingClub subset. Check data dictionary for feature meanings."
}
save_json(model_card, os.path.join(OUTDIR, "model_card.json"))
print("Saved model_card.json")

fastapi_code = f'''from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load(r"{best_model_path}")

@app.get("/")
def root():
    return {{\"status\":\"ok\",\"model\":\"{best_name}\"}}

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    # Apply the loaded pipeline for preprocessing and prediction
    proba = model.predict_proba(df)[:, 1]
    return {{\"pred_proba\": float(proba[0])}}
'''

fastapi_path = os.path.join(OUTDIR, "app.py")
with open(fastapi_path, "w") as f:
    f.write(fastapi_code)
print("Wrote FastAPI template to", fastapi_path)

print("\n--- Done ---")
print("Outputs written to:", OUTDIR)
print("- eval_results.json, psi_values.json, model_card.json")
print(f"- psi_alert.py (run: python {psi_alert_path} train.csv new.csv)")
print(f"- FastAPI template (run: uvicorn {os.path.basename(fastapi_path).replace('.py','')}:app --reload)")

# Feature Importance
# Access feature importances from the best Random Forest model (rf_clf)
# rf_clf is the best_estimator_ from RandomizedSearchCV which is a RandomForestClassifier
feature_importances = rf_clf.feature_importances_

# Get feature names from X_train
feature_names = X_train.columns

# Create a Pandas Series for better handling and sorting
importance_series = pd.Series(feature_importances, index=feature_names)

# Sort features by importance in descending order and get the top 20
top_20_features = importance_series.nlargest(20)
all_features = importance_series.nlargest(48)

print("Top 10 Most Influential Features:")
display(top_20_features)

# Optional: Visualize top 20 features
plt.figure(figsize=(10, 6))
sns.barplot(x=top_20_features.values, y=top_20_features.index, palette='plasma')
plt.title('Top 20 Most Influential Features')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

print("--- Calculating Feature-Target Correlations ---")

# Ensure 'loan_resampled' is the DataFrame containing the balanced data
# and 'bad_status' is the target variable

# Calculate correlations with the target variable ('bad_status')
# Use the 'loan_resampled' DataFrame which contains the balanced data with new features
# and is where the target variable 'bad_status' still exists as a column.
# If you've been working with X_res and y_res separately, you'd merge them first.

# For this step, I will assume 'loan_resampled' as the source for correlation calculation,
# as it is the DataFrame before train-test split and scaling, ensuring original values are used
# for correlation.

# First, let's reconstruct a full dataframe from X_res and y_res for correlation analysis.
# This is important because X_train/X_test are scaled, which can affect correlation values.
# The 'loan_resampled' dataframe should contain the preprocessed, balanced data.
# If `loan_resampled` is no longer available in the current scope, we will create it from X_res and y_res.

if 'loan_resampled' not in locals() or loan_resampled.empty:
    # If loan_resampled was not explicitly saved, recreate it from X_train and y_train
    # Ensure X_train and y_train are the SMOTEd and scaled versions for consistent correlation analysis
    loan_resampled = X_train.copy()
    loan_resampled['bad_status'] = y_train

correlations = loan_resampled.corr(numeric_only=True)['bad_status'].sort_values(ascending=False)

print("Top 20 Features Correlated with 'bad_status':")
display(correlations.head(20))

print("Bottom 20 Features Correlated with 'bad_status':")
display(correlations.tail(20))

# Visualize correlations using heatmaps
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(correlations.head(20).to_frame(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Top 20 Positive Correlations with bad_status')
plt.yticks(rotation=0)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(correlations.tail(20).to_frame(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Bottom 20 Negative Correlations with bad_status')
plt.yticks(rotation=0)
plt.show()

# ROC AUC Curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'r--', label='Random (AUC = 0.50)')

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        # Handle CalibratedClassifierCV wrapper
        if isinstance(model, CalibratedClassifierCV):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba (e.g., some linear models with decision_function)
        proba_raw = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-proba_raw)) # Sigmoid to convert to probabilities
        y_pred_proba = proba

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
