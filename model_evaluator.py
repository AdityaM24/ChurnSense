import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_score, recall_score
)
from data.feature_engineering import create_features


def evaluate_model(model_path="models/best_model.pkl",
                   data_path="data/cleaned_churn_data.csv"):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"F1-Score  : {f1_score(y, y_pred):.4f}")
    print(f"AUC-ROC   : {roc_auc_score(y, y_prob):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    evaluate_model()
