import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mode

# ---- Load models and preprocessing ----
svm = joblib.load("svm.pkl")          # SVC
lr = joblib.load("lr_model.pkl")      # LogisticRegression
rf = joblib.load("rf_model.pkl")      # RandomForestClassifier
knn = joblib.load("knn_model.pkl")    # KNeighborsClassifier

scaler = joblib.load("scaler.pkl")    # StandardScaler
imputer = joblib.load("imputer.pkl")  # SimpleImputer
encoder = joblib.load("label_encoder.pkl")  # LabelEncoder

models = [svm, lr, rf, knn]
model_names = ["SVM", "Logistic Regression", "Random Forest", "kNN"]

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
    # Create DataFrame
    X = pd.DataFrame([inputs], columns=feature_names)

    # ---- Preprocessing outside the model ----
    X_imp = imputer.transform(X)      # Handle missing values
    X_scaled = scaler.transform(X_imp)  # Scale features

    st.subheader("Individual Model Predictions")

    preds = []
    for name, model in zip(model_names, models):
        pred = model.predict(X_scaled)[0]
        preds.append(pred)
        st.write(f"{name}: {encoder.inverse_transform([pred])[0]}")

    preds = np.array(preds)

    # ---- Hard Voting ----
    final = mode(preds, keepdims=False).mode
    st.subheader("Final Verdict")
    st.write(encoder.inverse_transform([final])[0])

    # ---- Confidence ----
    confidence = np.sum(preds == final) / len(preds)
    st.write(f"Confidence: {confidence * 100:.0f}%")
