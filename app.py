# app.py
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

# Fungsi untuk preprocessing input
def preprocess_input(input_df):
    """Preprocess input sama seperti saat training"""
    df_processed = input_df.copy()
    
    # Mapping untuk binary features
    binary_mapping = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'SeniorCitizen': {'No': 0, 'Yes': 1}
    }
    
    # Apply binary mapping
    for col, mapping in binary_mapping.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # One-hot encoding untuk features dengan multiple categories
    categorical_features = ['InternetService', 'Contract', 'PaymentMethod']
    
    for col in categorical_features:
        if col in df_processed.columns:
            # Buat dummies
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            # Tambahkan ke dataframe
            df_processed = pd.concat([df_processed, dummies], axis=1)
            # Hapus kolom asli
            df_processed = df_processed.drop(col, axis=1)
    
    # Set default values untuk features lain yang mungkin dibutuhkan model
    # Sesuaikan dengan features yang digunakan saat training
    all_possible_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Tambahkan missing features dengan nilai 0
    for feature in all_possible_features:
        if feature not in df_processed.columns:
            df_processed[feature] = 0
    
    # Pastikan urutan kolom sesuai dengan training
    df_processed = df_processed.reindex(columns=all_possible_features, fill_value=0)
    
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
    payment_method = st.selectbox('Metode Pembayaran', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])

# Button untuk prediksi
if st.button('🔮 Prediksi Churn', type='primary'):
    if model is None:
        st.error("Model tidak tersedia. Pastikan model telah di-load.")
    else:
        # Buat dataframe dari input
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
        
        # Tampilkan input data
        st.subheader("📋 Data Input")
        st.dataframe(input_data)
        
        # Preprocess input
        with st.spinner('Memproses data...'):
            processed_data = preprocess_input(input_data)
            
            # Scaling jika scaler tersedia
            if scaler is not None:
                processed_data_scaled = scaler.transform(processed_data)
            else:
                processed_data_scaled = processed_data
        
        # Prediksi
        with st.spinner('Memprediksi...'):
            try:
                # Prediksi
                prediction = model.predict(processed_data_scaled)
                prediction_proba = model.predict_proba(processed_data_scaled)
                
                # Tampilkan hasil
                st.subheader("🎯 Hasil Prediksi")
                
                # Create columns for results
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    if prediction[0] == 1:
                        st.error("## ⚠️ CHURN")
                        st.metric("Status", "Pelanggan akan BERHENTI")
                    else:
                        st.success("## ✅ TIDAK CHURN")
                        st.metric("Status", "Pelanggan akan BERTAHAN")
                
                with result_col2:
                    st.metric("Probabilitas Churn", f"{prediction_proba[0][1]:.2%}")
                    st.metric("Probabilitas Tidak Churn", f"{prediction_proba[0][0]:.2%}")
                
                with result_col3:
                    # Confidence level
                    confidence = max(prediction_proba[0])
                    st.metric("Tingkat Kepercayaan", f"{confidence:.2%}")
                
                # Visualisasi probabilitas
                st.subheader("📊 Probabilitas Prediksi")
                
                prob_data = pd.DataFrame({
                    'Kelas': ['Tidak Churn', 'Churn'],
                    'Probabilitas': [prediction_proba[0][0], prediction_proba[0][1]]
                })
                
                # Bar chart
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(prob_data['Kelas'], prob_data['Probabilitas'], 
                             color=['#2ecc71', '#e74c3c'])
                ax.set_ylabel('Probabilitas')
                ax.set_title('Probabilitas Prediksi Churn')
                ax.set_ylim([0, 1])
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1%}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Rekomendasi
                st.subheader("💡 Rekomendasi")
                
                if prediction[0] == 1:
                    st.warning("""
                    **⚠️ PELANGGAN BERISIKO CHURN!**
                    
                    **Rekomendasi Tindakan:**
                    1. **Tawarkan promo khusus** untuk pelanggan yang ingin berhenti
                    2. **Hubungi pelanggan** untuk menanyakan alasan ketidakpuasan
                    3. **Analisis penyebab** churn berdasarkan data pelanggan
                    4. **Tingkatkan layanan** di area yang menjadi keluhan
                    5. **Berikan insentif** untuk tetap berlangganan
                    
                    **Faktor Risiko Tinggi:**
                    - Lama berlangganan pendek (< 12 bulan)
                    - Kontrak bulanan
                    - Biaya bulanan tinggi
                    - Tanpa layanan tambahan
                    """)
                else:
                    st.success("""
                    **✅ PELANGGAN LOYAL**
                    
                    **Rekomendasi Pemeliharaan:**
                    1. **Pertahankan kualitas layanan** yang sudah baik
                    2. **Tawarkan upgrade layanan** untuk meningkatkan engagement
                    3. **Berikan reward** untuk loyalitas pelanggan
                    4. **Monitor kepuasan** secara berkala
                    5. **Cross-sell produk/layanan** tambahan
                    
                    **Faktor Loyalitas:**
                    - Lama berlangganan panjang
                    - Kontrak tahunan
                    - Menggunakan multiple services
                    - Pembayaran otomatis
                    """)
                
                # Detail fitur penting
                st.subheader("🔍 Analisis Faktor Pengaruh")
                
                # Jika model Random Forest, tampilkan feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Fitur': processed_data.columns,
                        'Importance': model.feature_importances_
                    })
                    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.barh(feature_importance['Fitur'], feature_importance['Importance'])
                    ax2.set_xlabel('Tingkat Penting')
                    ax2.set_title('10 Fitur Paling Penting untuk Prediksi')
                    st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
                st.info("Pastikan semua input telah diisi dengan benar.")

# Bagian informasi dataset
with st.expander("📚 Informasi Dataset"):
    st.markdown("""
    **Telco Customer Churn Dataset** berisi informasi tentang pelanggan perusahaan telekomunikasi.
    
    **Variabel Target:** `Churn` (Yes/No)
    
    **Fitur Utama:**
    - **Demografi:** gender, SeniorCitizen, Partner, Dependents
    - **Layanan:** PhoneService, InternetService, OnlineSecurity, TechSupport, dll.
    - **Akuntansi:** tenure, Contract, PaperlessBilling, PaymentMethod
    - **Biaya:** MonthlyCharges, TotalCharges
    
    **Jumlah Data:** 7,043 records
    """)

# Footer
st.markdown("---")
st.caption("""
Proyek UAS Bengkel Koding Data Science - Universitas Dian Nuswantoro  
Deployed dengan Streamlit Cloud | Model: Random Forest Classifier

""")
