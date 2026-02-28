"""
ChurnSense Dashboard
Author : Aditya Mahale
About  : Streamlit dashboard for the churn prediction system.
         Shows KPI cards, EDA charts, a live prediction form, and a
         SHAP explanation chart for every prediction — built so a
         retention team can use it without touching any code.
"""
import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data.feature_engineering import create_features

st.set_page_config(
    page_title="ChurnSense | by Aditya Mahale",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .badge-high   { background:#e53e3e; color:#fff; padding:3px 10px; border-radius:10px; font-weight:600; }
  .badge-medium { background:#dd6b20; color:#fff; padding:3px 10px; border-radius:10px; font-weight:600; }
  .badge-low    { background:#38a169; color:#fff; padding:3px 10px; border-radius:10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_churn_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_resource
def load_shap():
    path = "models/shap_explainer.pkl"
    return joblib.load(path) if os.path.exists(path) else None

df = load_data()
model = load_model()
shap_data = load_shap()

# Sidebar
with st.sidebar:
    st.markdown("### Filters")
    contract_filter = st.multiselect(
        "Contract Type",
        options=df["Contract"].unique(),
        default=df["Contract"].unique()
    )
    tenure_range = st.slider(
        "Tenure (months)",
        int(df["tenure"].min()), int(df["tenure"].max()),
        (0, int(df["tenure"].max()))
    )

fdf = df[df["Contract"].isin(contract_filter) & df["tenure"].between(*tenure_range)]

# Header
st.title("📉 ChurnSense")
st.caption("Telecom Customer Churn Prediction · Aditya Mahale")
st.divider()

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(fdf):,}")
c2.metric("Churn Rate", f"{fdf['Churn'].mean():.1%}")
c3.metric("Retained", f"{(fdf['Churn']==0).sum():,}")
c4.metric("At Risk", f"{(fdf['Churn']==1).sum():,}")

st.divider()

# EDA: churn by contract (most actionable view)
st.subheader("Churn by Contract Type")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
fig.patch.set_facecolor("#0f1117")

# Churn by contract
plan_churn = fdf.groupby("Contract")["Churn"].mean().reset_index()
ax1 = axes[0]
ax1.set_facecolor("#1a1a2e")
bars = ax1.bar(plan_churn["Contract"], plan_churn["Churn"] * 100,
               color=["#e53e3e", "#dd6b20", "#38a169"], edgecolor="none", width=0.5)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", color="white", fontsize=9)
ax1.set_ylabel("Churn Rate (%)", color="white")
ax1.tick_params(colors="white")
ax1.set_title("Churn Rate by Contract", color="white", fontsize=11)
ax1.set_facecolor("#1a1a2e")

# Monthly charges distribution
ax2 = axes[1]
ax2.set_facecolor("#1a1a2e")
for churn_val, color, label in [(0, "#38a169", "Retained"), (1, "#e53e3e", "Churned")]:
    subset = fdf[fdf["Churn"] == churn_val]["MonthlyCharges"]
    ax2.hist(subset, bins=25, alpha=0.65, color=color, label=label, edgecolor="none")
ax2.set_xlabel("Monthly Charges ($)", color="white")
ax2.set_ylabel("Count", color="white")
ax2.tick_params(colors="white")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax2.set_title("Monthly Charges by Outcome", color="white", fontsize=11)

plt.tight_layout()
st.pyplot(fig)

st.divider()

# Prediction form
st.subheader("Predict Churn Risk")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure          = st.slider("Tenure (months)", 0, 100, 12)
        contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        payment_method   = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col3:
        streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        senior_citizen   = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner          = st.selectbox("Partner", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner, "Dependents": "No",
        "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }])

    try:
        input_eng = create_features(input_df)
        proba     = float(model.predict_proba(input_eng)[0, 1])
        risk      = "High" if proba > 0.7 else ("Medium" if proba > 0.4 else "Low")
        badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[risk]

        st.divider()
        r1, r2 = st.columns(2)
        r1.metric("Churn Probability", f"{proba:.1%}")
        r2.markdown(f"**Risk Level:** <span class='{badge_cls}'>{risk}</span>", unsafe_allow_html=True)
        st.progress(proba)

        if shap_data:
            st.markdown("**Top factors driving this prediction (SHAP)**")
            try:
                pre = model.named_steps["pre"]
                X_t = pre.transform(input_eng)
                explainer_obj = shap_data["explainer"]
                feat_names    = shap_data["feature_names"]

                shap_vals = explainer_obj.shap_values(X_t)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]

                top_n   = 10
                top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
                top_vals = sv[top_idx]
                top_labs = [feat_names[i] for i in top_idx]

                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                colors_shap = ["#e53e3e" if v > 0 else "#38a169" for v in top_vals]
                ax.barh(top_labs[::-1], top_vals[::-1], color=colors_shap[::-1], edgecolor="none")
                ax.axvline(0, color="white", linewidth=0.6)
                ax.set_xlabel("SHAP value (red = higher churn risk)", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                ax.set_title("Feature Contributions", color="white", fontsize=11)
                st.pyplot(fig)
            except Exception as se:
                st.warning(f"SHAP unavailable: {se}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.caption("ChurnSense · Aditya Mahale · IBM Telco Dataset · Logistic Regression + Random Forest + XGBoost + SHAP")
