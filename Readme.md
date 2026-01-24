# Task 1 | Credit Scoring System

## Overview
This project implements a **complete credit scoring system** to predict an individual's creditworthiness based on their financial history. Using the **UCI German Credit dataset**, I built a **robust machine learning pipeline** covering preprocessing, feature engineering, model training, evaluation, visualization, and sample predictions.

The system identifies potential credit risks and provides actionable insights for financial decision-making.

---

## Dataset
- **Source:** [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))  
- **Columns:** 21 features including financial, personal, and demographic information.  
- **Target:** `Creditability` (0 = Good Credit, 1 = Bad Credit)  
- **Note:** Original dataset was in raw `.data` format; preprocessing was required for proper ingestion.

**Key Features:**
- Numerical: `Duration_in_month`, `Credit_amount`, `Age`, etc.  
- Categorical: `Purpose`, `Housing`, `Job`, `Credit_history`, etc.  

---

## Project Steps

1. **Data Loading & Cleaning**
   - Loaded dataset and renamed columns.
   - Mapped target values (`1 -> 0`, `2 -> 1`) to define good and bad credit.

2. **Feature Engineering**
   - Separated numerical and categorical features.
   - Created interaction and ratio features where relevant.
   - Scaled numerical features and applied One-Hot Encoding to categorical features.

3. **Train-Test Split & Preprocessing**
   - 80-20 train-test split with stratification on the target.
   - Used `ColumnTransformer` to combine scaling and encoding.

4. **Model Training**
   - Trained multiple machine learning models:  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - Gradient Boosting
   - Pipelines ensured preprocessing was applied consistently.

5. **Evaluation & Metrics**
   - Measured performance using:  
     - Accuracy  
     - Precision  
     - Recall  
     - F1-Score  
     - ROC-AUC
   - Compared all models to identify the best performer.

6. **Visualization**
   - Plotted ROC curves for all models.
   - Plotted Precision-Recall curves.
   - Visualized feature importance (Random Forest).
   - Predicted probability distributions for Logistic Regression.

7. **Sample Predictions**
   - Tested model predictions on random samples from the test set.
   - Checked confidence levels and correctness of predictions.

8. **Model Export**
   - Saved the best-performing model (`Logistic Regression`) as a `.pkl` file for future use.

---

## Results
- **Best Model:** Logistic Regression  
- **Top Features (Random Forest):**  
  Identified key financial attributes contributing to credit risk.
- **Performance Metrics (Test Set):**
  - Accuracy: 0.78  
  - Precision: 0.76  
  - Recall: 0.71  
  - F1-Score: 0.73  
  - ROC-AUC: 0.84  

---
