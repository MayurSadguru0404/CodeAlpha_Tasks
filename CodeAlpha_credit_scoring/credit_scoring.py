import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

columns = [
    'Status_of_existing_checking_account','Duration_in_month','Credit_history','Purpose',
    'Credit_amount','Savings_account','Present_employment_since','Installment_rate',
    'Personal_status_and_sex','Other_debtors','Present_residence_since','Property',
    'Age','Other_installment_plans','Housing','Number_of_existing_credits','Job',
    'Number_of_people_liable','Telephone','Foreign_worker','Creditability']
df = pd.read_csv("german_credit.csv", sep=' ', header=None, names=columns)
df['Creditability'] = df['Creditability'].map({1:0, 2:1})  # 0 = Good, 1 = Bad

X = df.drop('Creditability', axis=1)
y = df['Creditability']
numerical_features = [
    'Duration_in_month','Credit_amount','Installment_rate','Present_residence_since',
    'Age','Number_of_existing_credits','Number_of_people_liable']
categorical_features = [col for col in X.columns if col not in numerical_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)}

results = {}
pipelines = {}
for name, clf in models.items():
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', clf)])
    pipeline.fit(X_train, y_train)
    pipelines[name] = pipeline

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)}
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['accuracy','precision','recall','f1_score','roc_auc']]
print("Model Performance Comparison:\n", comparison_df.round(4))

plt.figure(figsize=(10,6))
for name, pipeline in pipelines.items():
    y_proba = pipeline.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
for name, pipeline in pipelines.items():
    y_proba = pipeline.predict_proba(X_test)[:,1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall_vals, precision_vals, label=name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - All Models')
plt.legend()
plt.grid(True)
plt.show()

rf_pipeline = pipelines['Random Forest']
rf_model = rf_pipeline.named_steps['classifier']
ohe = rf_pipeline.named_steps['preprocessing'].named_transformers_['cat']
feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
print("\nTop 10 Features (Random Forest):")
print(feat_imp_df.head(10))

best_model = pipelines['Logistic Regression']
y_pred_proba = best_model.predict_proba(X_test)[:,1]
plt.figure(figsize=(8,5))
plt.hist(y_pred_proba[y_test==0], bins=20, alpha=0.6, label='Good Credit')
plt.hist(y_pred_proba[y_test==1], bins=20, alpha=0.6, label='Bad Credit')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution - Logistic Regression')
plt.legend()
plt.show()

sample_indices = np.random.choice(X_test.index, 5, replace=False)
for i, idx in enumerate(sample_indices,1):
    sample = X_test.loc[[idx]]
    actual = y_test.loc[idx]
    pred = best_model.predict(sample)[0]
    prob = best_model.predict_proba(sample)[0,1]
    status = "CORRECT" if pred==actual else "INCORRECT"
    print(f"Sample {i}: Predicted: {pred}, Actual: {actual}, Probability: {prob:.2%}, Status: {status}")

with open('credit_scoring_model.pkl','wb') as f:
    pickle.dump(best_model,f)
print("Model saved as 'credit_scoring_model.pkl'")
