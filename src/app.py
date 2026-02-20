from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

from data.feature_engineering import create_features

model = joblib.load("models/best_model.pkl")
shap_data = joblib.load("models/shap_explainer.pkl") if os.path.exists("models/shap_explainer.pkl") else None
explainer = shap_data["explainer"] if shap_data else None
feat_names = shap_data["feature_names"] if shap_data else None

app = FastAPI(title="Churn Prediction API", version="1.0.0")


class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
async def root():
    return {"service": "churn-prediction", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])
    df = create_features(df)

    proba = float(model.predict_proba(df)[0, 1])
    prediction = int(proba > 0.5)
    risk = "High" if proba > 0.7 else ("Medium" if proba > 0.4 else "Low")

    response = {
        "churn_prediction": prediction,
        "churn_probability": round(proba, 4),
        "risk_level": risk,
    }

    if explainer and feat_names:
        try:
            pre = model.named_steps["pre"]
            X_t = pre.transform(df)
            sv = explainer.shap_values(X_t)[0]
            top_indices = np.argsort(np.abs(sv))[::-1][:5]
            response["top_churn_drivers"] = [
                {"feature": feat_names[i], "shap_value": round(float(sv[i]), 4)}
                for i in top_indices
            ]
        except Exception:
            pass

    return response
