"""
ChurnSense — Feature Engineering
Author : Aditya Mahale
About  : Shared feature pipeline used at train time (model.py) and serve time
         (dashboard.py + src/app.py). Any new feature added here automatically
         flows through to predictions and SHAP explanations without extra wiring.

Features I engineered:
  tenure_group          — bucket tenure into lifecycle stages (new → loyal)
  charge_per_tenure     — monthly spend efficiency signal
  streaming_both        — double-streaming flag (engagement proxy)
  high_value_at_risk    — high spender on month-to-month contract
  service_count         — total add-ons subscribed (stickiness score)
  no_support_fiber      — fiber user with no tech support (highest-risk combo)
  monthly_vs_total_ratio — payment consistency / recency signal
"""
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bucket tenure into customer lifecycle stages
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
    )

    # Spend per month of tenure — high values indicate new high-spenders (risky)
    df['charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Customers using both streaming services tend to be more engaged
    df['streaming_both'] = (
        (df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')
    ).astype(int)

    # High-value customer locked into a flexible (easy-to-leave) contract
    df['high_value_at_risk'] = (
        (df['MonthlyCharges'] > df['MonthlyCharges'].median()) &
        (df['Contract'] == 'Month-to-month')
    ).astype(int)

    # Total add-on services (0–6) — more services = more switching cost = lower churn
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['service_count'] = sum(
        (df[col] == 'Yes').astype(int) for col in service_cols
    )

    # Fiber optic + no tech support = highest observed churn combination
    df['no_support_fiber'] = (
        (df['InternetService'] == 'Fiber optic') &
        (df['TechSupport'] != 'Yes')
    ).astype(int)

    # Low ratio = new customer or one whose charges have been rising fast
    df['monthly_vs_total_ratio'] = (
        df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    )

    return df