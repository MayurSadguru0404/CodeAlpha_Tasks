🩺 Disease Prediction using Machine Learning
📌 Project Overview

This project focuses on predicting diseases from structured medical data using machine learning classification techniques.
The implementation demonstrates a complete end-to-end ML pipeline, from data preprocessing to model evaluation and deployment readiness.

For demonstration, the pipeline is applied to the Breast Cancer Wisconsin dataset, a widely used benchmark in medical machine learning.

🎯 Objective

To predict the presence or absence of disease based on patient medical features using multiple classification models, while emphasizing interpretability, reliability, and evaluation metrics relevant to healthcare.

🧠 Key Concepts Covered

Medical data preprocessing

Handling missing values

Feature scaling

Binary classification

Model comparison using healthcare-relevant metrics

Model interpretability

Production-ready ML pipeline

📊 Dataset Used

Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository / Kaggle

Total samples: 569

Features: 30 numerical medical attributes

Target:

0 → Benign

1 → Malignant

The same pipeline can be reused for other medical datasets such as Heart Disease and Diabetes with minimal changes.

⚙️ Technologies & Libraries

Python

NumPy

Pandas

Matplotlib & Seaborn

Scikit-learn

XGBoost

🏗️ Machine Learning Pipeline

Data Loading

Load medical dataset from local CSV

Target Encoding

Convert diagnosis labels into binary format

Train-Test Split

Stratified split to preserve class distribution

Missing Value Handling

Mean imputation using SimpleImputer

Feature Scaling

Standardization using StandardScaler

Model Training

Logistic Regression (baseline)

Support Vector Machine (SVM)

Random Forest

XGBoost

Model Evaluation

Accuracy

Precision

Recall

F1-Score

ROC-AUC (primary metric for medical ML)

Model Selection

Best model selected based on ROC-AUC

Pipeline Creation

Imputer + Scaler + Model saved for reuse

📈 Results Summary
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.9649	0.975	0.9286	0.9512	0.996
SVM	0.9737	1.000	0.9286	0.9630	0.9947
Random Forest	0.9737	1.000	0.9286	0.9630	0.9929
XGBoost	0.9737	1.000	0.9286	0.9630	0.9940

🔹 Best Model: Logistic Regression
🔹 Reason: Highest ROC-AUC with strong interpretability

🩺 Why ROC-AUC Matters in Healthcare

In medical diagnosis:

Recall ensures fewer missed disease cases

ROC-AUC measures overall model reliability

Interpretability is often preferred over unnecessary complexity

This project prioritizes clinical relevance over raw accuracy.

💾 Saved Model

A production-ready pipeline is saved as:

disease_prediction_pipeline.pkl


This pipeline includes:

Missing value handling

Feature scaling

Trained classification model

It can be directly used for new patient predictions.

🚀 How to Run the Project

Clone the repository

Install dependencies

pip install -r requirements.txt


Run the project

python main.py

🔮 Future Enhancements

Extend pipeline to Heart Disease and Diabetes datasets

Hyperparameter tuning

Web or API deployment

Explainability tools (SHAP / LIME)

👤 Author

Mayur Sadguru
