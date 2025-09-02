import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("log_reg_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Credit Card Default Prediction App")

# Collect only important inputs from user
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", 1000, 1000000, 20000, step=1000)
sex = st.selectbox("Sex", [1, 2])  # 1=male, 2=female
education = st.selectbox("Education", [1, 2, 3, 4])  # categories from dataset
marriage = st.selectbox("Marriage", [1, 2, 3])  # categories
age = st.slider("Age", 18, 100, 30)

# For simplicity, we set remaining features = 0
other_features = {
    'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
    'BILL_AMT1': 0, 'BILL_AMT2': 0, 'BILL_AMT3': 0, 'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
    'PAY_AMT1': 0, 'PAY_AMT2': 0, 'PAY_AMT3': 0, 'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0
}

# Put all features together in correct order
input_dict = {
    'LIMIT_BAL': limit_bal,
    'SEX': sex,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'AGE': age,
    **other_features
}

# Convert to DataFrame with same column order as training
feature_order = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                 'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                 'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

input_df = pd.DataFrame([input_dict], columns=feature_order)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Default Probability"):
    prob = model.predict_proba(input_scaled)[0][1]
    st.write(f"🔮 Probability of Default: **{prob:.2f}**")

    if prob > 0.5:
        st.error("⚠️ High Risk: Customer likely to default. Consider rejecting the loan/credit.")
    else:
        st.success("✅ Low Risk: Customer likely to repay. Safe to approve credit.")
