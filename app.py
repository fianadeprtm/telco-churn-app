import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Setup
st.set_page_config(page_title="Prediksi Churn", layout="centered")
st.title("📊 Prediksi Churn Pelanggan Telco")
st.write("Aplikasi untuk memprediksi apakah pelanggan akan berhenti berlangganan")

# Load model
try:
    model = joblib.load('best_churn_model.pkl')
    st.success("✅ Model berhasil dimuat")
except:
    st.error("❌ Gagal memuat model")
    model = None

# Input form
st.header("Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Lama Berlangganan (bulan)", 0, 72, 12)
    monthly = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 70.0)
    total = st.number_input("Total Biaya ($)", 0.0, 10000.0, 1000.0)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])

with col2:
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Memiliki Partner", ["No", "Yes"])
    dependents = st.selectbox("Memiliki Tanggungan", ["No", "Yes"])
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])

# Predict button
if st.button("🔮 Prediksi") and model is not None:
    # Prepare input (sederhana)
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'Partner_Yes': 1 if partner == "Yes" else 0,
        'Dependents_Yes': 1 if dependents == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
    }])
    
    # Add missing columns with 0
    for col in ['PhoneService_Yes', 'InternetService_Fiber optic', 
                'InternetService_No', 'PaymentMethod_Credit card (automatic)',
                'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                'PaperlessBilling_Yes']:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    # Display result
    st.header("Hasil Prediksi")
    
    if prediction == 1:
        st.error(f"## ⚠️ CHURN ({proba[1]:.1%})")
        st.write("Pelanggan berisiko berhenti berlangganan")
    else:
        st.success(f"## ✅ TIDAK CHURN ({proba[0]:.1%})")
        st.write("Pelanggan kemungkinan akan bertahan")

st.markdown("---")
st.caption("Proyek UAS - Bengkel Koding Data Science UDINUS")
