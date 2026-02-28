"""
ChurnSense — Telecom Customer Churn Prediction
Author: Aditya Mahale
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_predict, RandomizedSearchCV
)
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score
)
from xgboost import XGBClassifier

from data.feature_engineering import create_features

warnings.filterwarnings("ignore")

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
PROCESSED_PATH = "data/cleaned_churn_data.csv"
MODEL_DIR      = "models"
REPORT_DIR     = "reports/figures"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


def load_and_clean(url):
    print("Loading dataset...")
    df = pd.read_csv(url)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    print(f"Rows: {len(df)} | Churn rate: {df['Churn'].mean():.1%}")
    return df


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def best_threshold(model, X_train, y_train, cv):
    """Find optimal decision threshold via OOF precision-recall curve."""
    oof_probs = cross_val_predict(model, X_train, y_train,
                                  cv=cv, method="predict_proba")[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_train, oof_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    return float(thresholds[np.argmax(f1s[:-1])])


def train_models(X_train, y_train, preprocessor):
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    spw = (y_train == 0).sum() / (y_train == 1).sum()  # for XGBoost imbalance
    models = {}

    print("\nTraining Logistic Regression...")
    lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                   class_weight="balanced", random_state=42))
    ])
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = (lr, best_threshold(lr, X_train, y_train, cv))
    print(f"  Threshold: {models['LogisticRegression'][1]:.2f}")

    print("Training Random Forest...")
    rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    models["RandomForest"] = (rf, best_threshold(rf, X_train, y_train, cv))
    print(f"  Threshold: {models['RandomForest'][1]:.2f}")

    print("Training XGBoost (RandomizedSearchCV)...")
    param_dist = {
        "clf__n_estimators":     [200, 300, 400, 500, 600],
        "clf__learning_rate":    [0.01, 0.02, 0.05, 0.1],
        "clf__max_depth":        [3, 4, 5, 6],
        "clf__min_child_weight": [1, 3, 5, 7, 10],
        "clf__gamma":            [0, 0.05, 0.1, 0.2, 0.3],
        "clf__subsample":        [0.7, 0.8, 0.85, 0.9],
        "clf__colsample_bytree": [0.6, 0.7, 0.75, 0.8],
        "clf__reg_alpha":        [0, 0.05, 0.1, 0.5],
        "clf__reg_lambda":       [1.0, 1.5, 2.0, 3.0],
        "clf__scale_pos_weight": [1.0, float(np.sqrt(spw)), float(spw)],
    }
    xgb_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1))
    ])
    search = RandomizedSearchCV(
        xgb_pipe, param_distributions=param_dist,
        n_iter=40, scoring="roc_auc",
        cv=cv, random_state=42, n_jobs=-1, refit=True, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV-AUC: {search.best_score_:.4f}")

    xgb = search.best_estimator_
    models["XGBoost"] = (xgb, best_threshold(xgb, X_train, y_train, cv))
    print(f"  Threshold: {models['XGBoost'][1]:.2f}")

    return models


def evaluate(models, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nModel Results (Test Set + 5-Fold CV AUC)")
    print(f"{'Model':<22} {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUC':>7}  {'CV-AUC':>8}")
    print("-" * 72)

    best_name, best_score, best_model = None, 0, None
    best_auc_val, best_f1_val = 0, 0

    for name, (model, threshold) in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        cv_probs = cross_val_predict(model, X_train, y_train,
                                     cv=cv, method="predict_proba")[:, 1]
        cv_auc = roc_auc_score(y_train, cv_probs)

        print(f"{name:<22} {acc:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {auc:>7.3f}  {cv_auc:>8.3f}")

        score = 0.5 * auc + 0.5 * cv_auc + 0.3 * f1
        if score > best_score:
            best_score, best_name, best_model = score, name, model
            best_auc_val, best_f1_val = auc, f1

    print(f"\nBest model: {best_name}  |  AUC={best_auc_val:.3f}  F1={best_f1_val:.3f}")
    return best_name, best_model


def generate_shap(models, X_test):
    """Generate SHAP summary using XGBoost (preferred) or Random Forest."""
    shap_model, shap_name = None, None
    for name in ["XGBoost", "RandomForest"]:
        if name in models:
            shap_model, shap_name = models[name][0], name
            break
    if shap_model is None:
        return None, None

    print(f"\nGenerating SHAP values ({shap_name})...")
    pre = shap_model.named_steps["pre"]
    clf = shap_model.named_steps["clf"]

    X_t = pre.transform(X_test)
    num_cols  = X_test.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols  = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
    ohe_names = pre.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    feat_names = num_cols + ohe_names

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_t)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_t, feature_names=feat_names, plot_type="bar", show=False)
    plt.title(f"Feature Importance — SHAP ({shap_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_summary.png", dpi=150)
    plt.close()

    joblib.dump({"explainer": explainer, "feature_names": feat_names},
                f"{MODEL_DIR}/shap_explainer.pkl")
    print(f"SHAP explainer saved → {MODEL_DIR}/shap_explainer.pkl")
    return explainer, feat_names


if __name__ == "__main__":
    df = load_and_clean(RAW_DATA_URL)
    df = create_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned data saved → {PROCESSED_PATH}")

    X, y = df.drop("Churn", axis=1), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    models       = train_models(X_train, y_train, preprocessor)
    best_name, best_model = evaluate(models, X_train, y_train, X_test, y_test)

    for name, (m, _) in models.items():
        joblib.dump(m, f"{MODEL_DIR}/{name.lower()}_model.pkl")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    print(f"\nAll models saved → {MODEL_DIR}/")

    generate_shap(models, X_test)
    print("\nDone.")
