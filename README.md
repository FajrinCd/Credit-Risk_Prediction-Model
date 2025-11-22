# Credit Risk Modeling — LendingClub Dataset (2007–2014)

---

This project provides an end-to-end pipeline for building a **Credit Risk Prediction Model** using LendingClub loan data from 2007–2014.
The workflow includes **data cleaning, feature engineering, model training, evaluation, explainability, PSI monitoring**, and a **FastAPI deployment template**.

---


## Key Features

* Full preprocessing pipeline (cleaning, encoding, imputation)
* Class imbalance handling using **SMOTE**
* Engineered credit risk–specific features:

  * `credit_age`
  * `loan_to_income_ratio`
  * `installment_to_income_ratio`
  * `revolving_burden_index`
  * `payment_progress_ratio`, etc.
* Models trained:

  * Logistic Regression
  * Random Forest (with RandomizedSearchCV)
  * XGBoost
* Model calibration (Isotonic)
* SHAP explainability for tree-based models
* PSI calculation for feature stability
* Automatic export:

  * `eval_results.json`
  * `psi_values.json`
  * `model_card.json`
  * `FastAPI` prediction service

---

## Output Files

Generated under the **output/** directory:

```
best_model_<name>.joblib
eval_results.json
psi_values.json
model_card.json
app.py   # FastAPI service
```

---

## Overview of the Pipeline

1. **Load & explore data**
2. **Data cleaning**

   * Drop irrelevant fields
   * Convert date fields to numeric age/month
   * Encode categorical variables
   * Handle missing values
3. **Feature engineering**
4. **SMOTE balancing**
5. **Train/Test split + scaling**
6. **Model training & evaluation**
7. **Feature importance + ROC Curve**
8. **SHAP explainer**
9. **PSI stability check**
10. **Model export + deployment template**

---

## FastAPI Deployment

Start service:

```bash
uvicorn app:app --reload
```

Example request:

```json
{
  "loan_amnt": 10000,
  "annual_inc": 55000,
  "int_rate": 14.5
}
```

Response:

```json
{
  "pred_proba": 0.27
}
```

---

## Notes

* The dataset is not included due to size constraints.
* SHAP is computed using up to 500 samples for performance.
* All features are fully numeric after preprocessing.

---

For questions or feedback, open an issue on GitHub or contact the maintainer: **[dgartup@gmail.com](mailto:dgartup@gmail.com)**
Happy coding! 🚀
