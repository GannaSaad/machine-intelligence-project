import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mode

# ---- Load models ----
svm = joblib.load("svm.pkl")          # SVC
lr = joblib.load("lr_model.pkl")      # LogisticRegression
rf = joblib.load("rf_model.pkl")      # RandomForestClassifier
knn = joblib.load("knn_model.pkl")    # KNeighborsClassifier
scaler = joblib.load("scaler.pkl")

models = [svm, lr, rf, knn]
model_names = ["SVM", "Logistic Regression", "Random Forest", "kNN"]

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

    preds = np.array(preds)

    # ---- Hard Voting ----
    final = mode(preds, keepdims=False).mode

    st.subheader("Final Verdict")
    st.write(labels[int(final)])

    # ---- Confidence ----
    confidence = np.sum(preds == final) / len(preds)
    st.write(f"Confidence: {confidence * 100:.0f}%")
