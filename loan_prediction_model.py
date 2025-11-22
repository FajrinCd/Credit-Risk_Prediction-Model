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

# Data Cleaning
# 'emp_length' Column
# Define a mapping for employment length to numerical values
emp_length_mapping = {
    '< 1 year': 0.5,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10,
    'unknown': np.nan # Handle 'unknown' from previous cleaning as NaN
}

# Ensure 'emp_length' is string type and handle NaN before mapping
loan['emp_length_cleaned'] = loan['emp_length'].astype(str).str.strip().str.lower().replace({'nan': 'unknown', '': 'unknown'})
loan['emp_length_num'] = loan['emp_length_cleaned'].map(emp_length_mapping)
# Convert to numeric, coercing any remaining errors to NaN
loan['emp_length_num'] = pd.to_numeric(loan['emp_length_num'], errors='coerce')

print(f"Converted 'emp_length' to numerical column 'emp_length_num': \n{loan['emp_length_num'].head()}")
loan.drop(['emp_length', 'emp_length_cleaned'], axis=1, inplace=True)

# 'term' Column
print("--- Processing 'term' column ---")
loan['term_num'] = loan['term'].str.replace('months','').astype(float)
print(f"Converted 'term' to numerical column 'term_num': \n{loan['term_num'].head()}")
loan.drop('term', axis=1, inplace=True)

# 'earliest_cr_line'
print("--- Processing 'earliest_cr_line' column ---")
print(loan['earliest_cr_line'].head())
# Correct column name 'earliest_cr_line' and format '%b-%y'
loan['earliest_cr_year'] = pd.to_datetime(loan['earliest_cr_line'], format='%b-%y', errors='coerce').dt.year
print(f"Converted 'earliest_cr_line' to numerical year 'earliest_cr_year': \n{loan['earliest_cr_year'].head()}")
loan.drop('earliest_cr_line', axis=1, inplace=True)

# ' issue_d'
print("--- Processing 'issue_d' column ---")
loan['issue_d_date'] = pd.to_datetime(loan['issue_d'], format='%b-%y', errors='coerce')
loan['issue_d_month'] = round(pd.to_datetime('2017-12-01') - loan['issue_d_date']).dt.days/30
print(f"Converted 'issue_d' to numerical month 'issue_d_month': \n{loan['issue_d_month'].head()}")
loan.drop(['issue_d', 'issue_d_date'], axis=1, inplace=True)

# 'last_pymnt_d'
print("--- Processing 'last_pymnt_d' column ---")
loan['last_pymnt_d_date'] = pd.to_datetime(loan['last_pymnt_d'], format='%b-%y', errors='coerce')
loan['last_pymnt_d_month'] = round(pd.to_datetime('2017-12-01') - loan['last_pymnt_d_date']).dt.days/30
print(f"Converted 'last_pymnt_d' to numerical month: \n{loan['last_pymnt_d_month'].head()}")
loan.drop(['last_pymnt_d', 'last_pymnt_d_date'],  axis=1, inplace=True)

# 'next_pymnt_d'
print("--- Processing 'next_pymnt_d' column ---")
loan['next_pymnt_d_date'] = pd.to_datetime(loan['next_pymnt_d'], format='%b-%y', errors='coerce')
loan['next_pymnt_d_month'] = round(pd.to_datetime('2017-12-01') - loan['next_pymnt_d_date']).dt.days/30
print(f"Converted 'next_pymnt_d' to numerical month: \n{loan['next_pymnt_d_month'].head()}")
loan.drop(['next_pymnt_d', 'next_pymnt_d_date'], axis=1, inplace=True)

# 'last_credit_pull_d'
print("--- Processing 'last_credit_pull_d' column ---")
loan['last_credit_pull_d_date'] = pd.to_datetime(loan['last_credit_pull_d'], format='%b-%y', errors='coerce')
loan['last_credit_pull_d_month'] = round(pd.to_datetime('2017-12-01') - loan['last_credit_pull_d_date']).dt.days/30
print(f"Converted 'last_credit_pull_d' to numerical month: \n{loan['last_credit_pull_d_month'].head()}")
loan.drop(['last_credit_pull_d', 'last_credit_pull_d_date'], axis=1, inplace=True)

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

for col in categorical_var:
    loan[col] = label.fit_transform(loan[col])

loan.head()

# Create New Feature
# 1. Create credit_age
loan['credit_age'] = 2017 - loan['earliest_cr_year']
print(f"'credit_age' created. First 5 values:\n{loan['credit_age'].head()}")

# 2. Create loan_to_income_ratio
# Add a small epsilon to annual_inc to avoid division by zero
epsilon = 1e-6
loan['loan_to_income_ratio'] = loan['loan_amnt'] / (loan['annual_inc'] + epsilon)
print(f"'loan_to_income_ratio' created. First 5 values:\n{loan['loan_to_income_ratio'].head()}")

# 3. Create installment_to_income_ratio
loan['installment_to_income_ratio'] = loan['installment'] / (loan['annual_inc'] + epsilon)
print(f"'installment_to_income_ratio' created. First 5 values:\n{loan['installment_to_income_ratio'].head()}")

print("New features added to DataFrame. Displaying head with new columns:")
loan.head()

# 4. Initialize a small constant epsilon to prevent division by zero
epsilon = 1e-6

# 5. Create revolving_burden_index
loan['revolving_burden_index'] = loan['revol_bal'] / (loan['annual_inc'] + epsilon)
print(f"'revolving_burden_index' created. First 5 values:\n{loan['revolving_burden_index'].head()}")

# 6. Create principal_repayment_ratio
loan['principal_repayment_ratio'] = loan['total_rec_prncp'] / (loan['loan_amnt'] + epsilon)
print(f"'principal_repayment_ratio' created. First 5 values:\n{loan['principal_repayment_ratio'].head()}")

# 7. Create payment_progress_ratio
loan['payment_progress_ratio'] = loan['total_pymnt'] / (loan['loan_amnt'] + epsilon)
print(f"'payment_progress_ratio' created. First 5 values:\n{loan['payment_progress_ratio'].head()}")

# 8. Create interest_payment_indicator
loan['interest_payment_indicator'] = loan['total_rec_int'] / (loan['total_pymnt'] + epsilon)
print(f"'interest_payment_indicator' created. First 5 values:\n{loan['interest_payment_indicator'].head()}")

# 9. Create funded_ratio
loan['funded_ratio'] = loan['funded_amnt'] / (loan['loan_amnt'] + epsilon)
print(f"'funded_ratio' created. First 5 values:\n{loan['funded_ratio'].head()}")

print("New features added to DataFrame. Displaying head with all new columns:")
loan.head()

# Handling Imbalance Data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

print("--- Handling Imbalanced Data ---")

# Separate features (X) and target (y)
X = loan.drop('bad_status', axis=1)
y = loan['bad_status']

print("Original target distribution:")
print(y.value_counts(normalize=True) * 100)

# Apply SMOTE for oversampling
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("\nTarget distribution after SMOTE oversampling:")
print(y_res.value_counts(normalize=True) * 100)

# Assign resampled data back to loan_resampled DataFrame (optional, for continuity)
# This creates a new DataFrame for the resampled data
loan_resampled = X_res.copy()
loan_resampled['bad_status'] = y_res

print(f"\nShape of original data: {loan.shape}")
print(f"Shape of resampled data: {loan_resampled.shape}")

# Train Test Split
from sklearn.model_selection import train_test_split

print("--- Performing Train-Test Split (with new features) ---")

# Split the resampled data into training and testing sets
# Using stratify=y_res to maintain the 50/50 class distribution in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

print("\nTarget distribution in y_train (percentage):")
print(y_train.value_counts(normalize=True) * 100)

print("\nTarget distribution in y_test (percentage):")
print(y_test.value_counts(normalize=True) * 100)

# Standardization
from sklearn.preprocessing import StandardScaler

# Identify numerical columns for scaling
numerical_cols = X_train.select_dtypes(include=np.number).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

X_train_scaled = pd.DataFrame(X_train, columns=numerical_cols)
X_test_scaled = pd.DataFrame(X_test, columns=numerical_cols)

print("Standardization complete for numerical features.")
print("First 5 rows of X_train after standardization:")
print(X_train.head())
print("\nFirst 5 rows of X_test after standardization:")
print(X_test.head())

# MODEL
# Logistic Regression (simple, interpretable)
log_clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
log_clf.fit(X_train, y_train)

# Random Forest with RandomizedSearchCV (light and faster)
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
search.fit(X_train, y_train)
rf_clf = search.best_estimator_
print("RF best params:", search.best_params_)

# XGBoost
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    verbosity=0,
    random_state=42
)

try:
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
except TypeError:
    print("Warning: early_stopping_rounds not supported, fitting without it.")
    xgb_clf.fit(X_train, y_train)

# Calibration (optional): calibrate RF probabilities (isotonic)
calibrated_rf = CalibratedClassifierCV(rf_clf, cv=2, method='isotonic')
calibrated_rf.fit(X_train, y_train)

# Evaluation
models = {
    "Logistic": log_clf,
    "RandomForest": calibrated_rf,
    "XGBoost": xgb_clf
}

eval_results = {}
for name, m in models.items():
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(X_test)[:, 1]
    else:
        proba_raw = m.decision_function(X_test)
        proba_raw = np.clip(proba_raw, -10, 10)
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

# SHAP
# Define feature_names for SHAP (all columns in X_train are features)
feature_names = X_train.columns.tolist()

# Explainability (SHAP) - only for tree models (RF/XGB)
try:
    import shap
    print("\nComputing SHAP (limited to 500 samples for speed)...")
    if best_name in ["RandomForest", "XGBoost"]:
        tree_model = best_model
        # If CalibratedClassifierCV is used, get the base estimator
        if hasattr(best_model, "base_estimator_"):
            tree_model = best_model.base_estimator_

        # Ensure tree_model is a valid type for TreeExplainer
        if isinstance(tree_model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(tree_model)
            max_samples = 500
            # Use X_test if available and large enough, otherwise X_train
            X_shap = X_test if X_test.shape[0] <= max_samples else X_test[:max_samples]
            shap_values = explainer.shap_values(X_shap)
            if feature_names and len(feature_names) == X_shap.shape[1]:
                shap.summary_plot(shap_values, X_shap, feature_names=feature_names, max_display=15)
            else:
                shap.summary_plot(shap_values, X_shap, max_display=15)
        else:
            print("SHAP TreeExplainer skipped: best model is a CalibratedClassifierCV wrapper around a non-tree-based model.")
    else:
        print("SHAP TreeExplainer skipped: best model not tree-based.")
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

# Use numerical_cols from previous step for PSI calculation
# All columns in X_train are numerical after label encoding and type conversions
psi_features = X_train.columns.tolist()[:10] # Use first 10 features for PSI example
psi_report_vals = {}
for f in psi_features:
    # Use X_train and X_test directly as they contain the processed data
    if f in X_train.columns and f in X_test.columns:
        psi_report_vals[f] = psi(X_train[f].dropna().values, X_test[f].dropna().values, buckets=10)
    else:
        psi_report_vals[f] = None

print("\nPSI values for sample features:")
print(psi_report_vals)
save_json(psi_report_vals, os.path.join(OUTDIR, "psi_values.json"))

# Path psi_alert.py
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

print("Top 20 Most Influential Features:")
display(top_20_features)

# Visualize top 20 features
plt.figure(figsize=(10, 6))
sns.barplot(x=top_20_features.values, y=top_20_features.index, palette='crest')
plt.title('Top 20 Most Influential Features')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
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
