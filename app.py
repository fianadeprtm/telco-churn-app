# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Prediksi Churn Pelanggan Telco",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Prediksi Churn Pelanggan Telco")
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan akan berhenti berlangganan (churn) 
berdasarkan karakteristik dan riwayat layanan mereka.
""")

# Sidebar untuk informasi
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.info("""
**Model Machine Learning:** Random Forest Classifier  
**Metrik Terbaik:** F1-Score  
**Dataset:** Telco Customer Churn
""")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_churn_model.pkl')
        return model
    except:
        st.error("Gagal memuat model. Pastikan file 'best_churn_model.pkl' tersedia.")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except:
        return None

model = load_model()
scaler = load_scaler()

# Fungsi preprocessing input
def preprocess_input(input_df):
    df_processed = input_df.copy()
    
    # Mapping binary features
    binary_mapping = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    # Apply binary mapping
    for col, mapping in binary_mapping.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Handle SeniorCitizen
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({'No': 0, 'Yes': 1})
    
    # One-hot encoding
    categorical_features = ['InternetService', 'Contract', 'PaymentMethod']
    
    for col in categorical_features:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed = df_processed.drop(col, axis=1)
    
    # Ensure all required features exist
    all_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Add missing features
    for feature in all_features:
        if feature not in df_processed.columns:
            df_processed[feature] = 0
    
    # Reorder columns
    df_processed = df_processed.reindex(columns=all_features, fill_value=0)
    
    return df_processed

# Input form
st.header("📝 Input Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input('Lama Berlangganan (bulan)', min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input('Biaya Bulanan ($)', min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input('Total Biaya ($)', min_value=0.0, max_value=10000.0, value=1000.0)

with col2:
    gender = st.selectbox('Jenis Kelamin', ['Female', 'Male'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Memiliki Partner', ['No', 'Yes'])
    dependents = st.selectbox('Memiliki Tanggungan', ['No', 'Yes'])

with col3:
    phone_service = st.selectbox('Layanan Telepon', ['No', 'Yes'])
    internet_service = st.selectbox('Layanan Internet', ['DSL', 'Fiber optic', 'No'])
    contract = st.selectbox('Jenis Kontrak', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Metode Pembayaran', 
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])

# Prediction button
if st.button('🔮 Prediksi Churn', type='primary'):
    if model is None:
        st.error("Model tidak tersedia.")
    else:
        # Create input dataframe
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'PhoneService': [phone_service],
            'InternetService': [internet_service],
            'Contract': [contract],
            'PaymentMethod': [payment_method],
            'PaperlessBilling': [paperless_billing]
        })
        
        st.subheader("📋 Data Input")
        st.dataframe(input_data)
        
        # Preprocess
        with st.spinner('Memproses data...'):
            processed_data = preprocess_input(input_data)
            
            if scaler is not None:
                processed_data_scaled = scaler.transform(processed_data)
            else:
                processed_data_scaled = processed_data
        
        # Predict
        with st.spinner('Memprediksi...'):
            try:
                prediction = model.predict(processed_data_scaled)
                prediction_proba = model.predict_proba(processed_data_scaled)
                
                st.subheader("🎯 Hasil Prediksi")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction[0] == 1:
                        st.error("## ⚠️ CHURN")
                    else:
                        st.success("## ✅ TIDAK CHURN")
                
                with col2:
                    st.metric("Probabilitas Churn", f"{prediction_proba[0][1]:.2%}")
                    st.metric("Probabilitas Tidak Churn", f"{prediction_proba[0][0]:.2%}")
                
                with col3:
                    confidence = max(prediction_proba[0])
                    st.metric("Tingkat Kepercayaan", f"{confidence:.2%}")
                
                # Visualization
                st.subheader("📊 Probabilitas Prediksi")
                prob_data = pd.DataFrame({
                    'Kelas': ['Tidak Churn', 'Churn'],
                    'Probabilitas': [prediction_proba[0][0], prediction_proba[0][1]]
                })
                st.bar_chart(prob_data.set_index('Kelas'))
                
                # Recommendations
                st.subheader("💡 Rekomendasi")
                if prediction[0] == 1:
                    st.warning("**Pelanggan berisiko CHURN!** Rekomendasi: tawarkan promo, hubungi untuk feedback.")
                else:
                    st.success("**Pelanggan loyal.** Pertahankan kualitas layanan.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Proyek UAS Bengkel Koding Data Science - Universitas Dian Nuswantoro")

