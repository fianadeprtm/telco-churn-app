import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('best_churn_model.pkl')
    except:
        st.error("Model tidak ditemukan")
        return None

@st.cache_resource 
def load_scaler():
    try:
        return joblib.load('scaler.pkl')
    except:
        return None

# Setup page
st.set_page_config(page_title="Prediksi Churn", layout="wide")
st.title("📊 Prediksi Churn Pelanggan")

# Load resources
model = load_model()
scaler = load_scaler()

# Input form
st.header("📝 Input Data")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Lama Berlangganan (bulan)", 0, 100, 12)
    monthly = st.number_input("Biaya Bulanan", 0.0, 200.0, 70.0)
    total = st.number_input("Total Biaya", 0.0, 10000.0, 1000.0)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Tanggungan", ["No", "Yes"])
    phone = st.selectbox("Layanan Telepon", ["No", "Yes"])
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])

# Preprocessing function
def prepare_input():
    # Create dataframe
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'gender': 0 if gender == "Female" else 1,
        'SeniorCitizen': 0 if senior == "No" else 1,
        'Partner': 0 if partner == "No" else 1,
        'Dependents': 0 if dependents == "No" else 1,
        'PhoneService': 0 if phone == "No" else 1,
        'InternetService_DSL': 1 if internet == "DSL" else 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,
        'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
    }
    
    # Add default values for other features
    features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'gender', 'Partner', 'Dependents', 'PhoneService', 
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'PaperlessBilling'
    ]
    
    df = pd.DataFrame([data])
    
    # Add missing columns with 0
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[features]

# Predict button
if st.button("Prediksi", type="primary") and model:
    input_data = prepare_input()
    
    # Scale if scaler exists
    if scaler:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.header("🎯 Hasil")
    
    if prediction == 1:
        st.error(f"CHURN (Probabilitas: {proba[1]:.1%})")
        st.warning("Pelanggan berisiko berhenti berlangganan")
    else:
        st.success(f"TIDAK CHURN (Probabilitas: {proba[0]:.1%})")
        st.info("Pelanggan kemungkinan akan bertahan")
    
    # Probability bar
    st.progress(float(proba[1]))
    st.caption(f"Tingkat churn: {proba[1]:.1%}")

st.markdown("---")
st.caption("Proyek UAS - Bengkel Koding Data Science")
