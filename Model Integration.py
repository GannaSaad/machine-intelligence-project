import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mode

# ---- Load models ----
svm = joblib.load("svm.pkl")
lr = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")  # if used

models = [svm, lr]
model_names = ["SVM", "Logistic Regression"]

labels = ['Away Win', 'Draw', 'Home Win']

feature_names = [
    'Half Time Home Goals',
    'Half Time Away Goals',
    'Home Shots on Target',
    'Away Shots on Target'
]

st.title("Football Match Outcome Predictor")

# ---- User Inputs ----
inputs = []
for f in feature_names:
    inputs.append(st.number_input(f, min_value=0, step=1))

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=feature_names)
    X_scaled = scaler.transform(X)

    st.subheader("Individual Model Predictions")
    preds = []
    for name, model in zip(model_names, models):
        pred = model.predict(X_scaled)[0]
        preds.append(pred)
        st.write(f"{name}: {labels[int(pred)]}")

    # ---- Final Voting ----
    final = mode(preds, keepdims=False).mode
    st.subheader("Final Verdict")
    st.write(labels[int(final)])

    # ---- Confidence (fraction of models agreeing) ----
    confidence = np.sum(preds == final) / len(models)
    st.write(f"Confidence: {confidence*100:.0f}%")
