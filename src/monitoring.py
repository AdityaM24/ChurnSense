"""
ChurnSense — Model Monitoring
Author : Aditya Mahale
About  : ModelMonitor tracks F1 and AUC over new data batches and fires
         an alert when either metric drops >15% relative to the training
         baseline — a lightweight but practical production health check.
         Logs are written to monitoring/performance_log.csv; alerts go
         to monitoring/alerts.log so they can be piped into any alerting
         system (Slack, PagerDuty, email, etc.).
"""
import os
import warnings
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score

from data.feature_engineering import create_features


class ModelMonitor:
    def __init__(self, model_path="models/best_model.pkl",
                 reference_path="data/cleaned_churn_data.csv"):
        self.model = joblib.load(model_path)
        ref = pd.read_csv(reference_path)
        X_ref = create_features(ref.drop("Churn", axis=1))
        y_ref = ref["Churn"]

        self.reference_metrics = {
            "f1_score": f1_score(y_ref, self.model.predict(X_ref)),
            "roc_auc": roc_auc_score(y_ref, self.model.predict_proba(X_ref)[:, 1]),
            "n": len(X_ref),
        }

        os.makedirs("monitoring", exist_ok=True)
        self._log_path = "monitoring/performance_log.csv"
        if os.path.exists(self._log_path):
            self.performance_log = pd.read_csv(self._log_path)
        else:
            self.performance_log = pd.DataFrame(
                columns=["timestamp", "dataset", "f1_score", "roc_auc", "sample_size"]
            )

    def log_performance(self, X, y_true, dataset_name):
        try:
            X_fe = create_features(X)
            y_pred = self.model.predict(X_fe)
            y_prob = self.model.predict_proba(X_fe)[:, 1]
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset_name,
                "f1_score": f1_score(y_true, y_pred),
                "roc_auc": roc_auc_score(y_true, y_prob),
                "sample_size": len(X_fe),
            }
            self.performance_log = pd.concat(
                [self.performance_log, pd.DataFrame([row])], ignore_index=True
            )
            self._check_drift(row)
            self.performance_log.to_csv(self._log_path, index=False)
            return row
        except Exception as e:
            warnings.warn(f"log_performance error: {e}")
            return None

    def _check_drift(self, current, threshold=0.15):
        f1_drop = (self.reference_metrics["f1_score"] - current["f1_score"]) / self.reference_metrics["f1_score"]
        auc_drop = (self.reference_metrics["roc_auc"] - current["roc_auc"]) / self.reference_metrics["roc_auc"]
        alerts = []
        if f1_drop > threshold:
            alerts.append(f"F1 degraded {f1_drop:.1%}")
        if auc_drop > threshold:
            alerts.append(f"AUC degraded {auc_drop:.1%}")
        if alerts:
            msg = "ALERT - Model drift: " + " | ".join(alerts)
            print(msg)
            with open("monitoring/alerts.log", "a") as fh:
                fh.write(f"{datetime.now()}: {msg}\n")

    def get_performance_history(self):
        return self.performance_log

    def generate_report(self):
        if self.performance_log.empty:
            return "No data yet."
        report = (
            f"Churn Model Monitoring Report\n"
            f"Generated: {datetime.now()}\n\n"
            f"Reference F1={self.reference_metrics['f1_score']:.3f} "
            f"AUC={self.reference_metrics['roc_auc']:.3f} n={self.reference_metrics['n']}\n\n"
            f"History:\n{self.performance_log.describe().to_string()}"
        )
        with open("monitoring/report.txt", "w") as fh:
            fh.write(report)
        return report
