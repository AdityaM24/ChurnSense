"""
ChurnSense — Scheduled Monitoring Runner
Author: Aditya Mahale
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from datetime import datetime
from src.monitoring import ModelMonitor


def main():
    print(f"[{datetime.now()}] ChurnSense monitoring run started")
    monitor  = ModelMonitor()
    data     = pd.read_csv("data/cleaned_churn_data.csv").sample(frac=0.2, random_state=None)
    X, y     = data.drop("Churn", axis=1), data["Churn"]
    metrics  = monitor.log_performance(X, y, f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}")
    if metrics:
        print(f"  F1={metrics['f1_score']:.3f}  AUC={metrics['roc_auc']:.3f}  n={metrics['sample_size']}")
    print(monitor.generate_report())


if __name__ == "__main__":
    main()
