import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve)
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data.csv")
df["target"] = df["diagnosis"].map({"M": 1, "B": 0})
df.drop(["id", "diagnosis"], axis=1, inplace=True)
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False)}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    print(f"\n{name}")
    print("Accuracy :", round(results[name]["accuracy"], 4))
    print("Precision:", round(results[name]["precision"], 4))
    print("Recall   :", round(results[name]["recall"], 4))
    print("F1-score :", round(results[name]["f1_score"], 4))
    print("ROC-AUC  :", round(results[name]["roc_auc"], 4))

sns.set_style("whitegrid")
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind="bar", figsize=(12, 6))
plt.title("Model Performance Comparison")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

n_features = models["Random Forest"].n_features_in_

# Align feature names with model
feature_names = X_train.columns[:n_features]

rf_importance = models["Random Forest"].feature_importances_

rf_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_importance
}).sort_values("importance", ascending=False).head(10)

plt.figure(figsize=(8, 6))
plt.barh(rf_df["feature"], rf_df["importance"])
plt.gca().invert_yaxis()
plt.title("Random Forest - Top 10 Features")
plt.xlabel("Importance")
plt.show()

best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
best_model = models[best_model_name]
print("\nBest Model:", best_model_name)
print("Best ROC-AUC:", round(results[best_model_name]["roc_auc"], 4))

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", best_model)])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "disease_prediction_pipeline.pkl")
print("\nPipeline saved as disease_prediction_pipeline.pkl")
