# app.py - VERSI SEDERHANA
"""
Streamlit App untuk Prediksi Churn Pelanggan
UAS Bengkel Koding Data Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="wide"
)

# Load model dan preprocessor
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except:
        st.error("⚠️ Model atau preprocessor tidak ditemukan!")
        st.info("Jalankan notebook `analysis.ipynb` terlebih dahulu untuk membuat model.")
        return None, None

model, preprocessor = load_models()

# Sidebar untuk navigasi (SEDERHANA)
st.sidebar.title("📊 Menu Navigasi")
st.sidebar.markdown("---")

# Pilihan halaman menggunakan selectbox
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["🏠 Beranda", "🔮 Prediksi", "📊 Analisis", "🤖 Model"]
)

st.sidebar.markdown("---")
st.sidebar.info("UAS Bengkel Koding Data Science")

# ============================================================
# Halaman 1: BERANDA
# ============================================================
if page == "🏠 Beranda":
    st.title("🏠 Prediksi Churn Pelanggan Telco")
    st.markdown("---")
    
    st.write("""
    ## Selamat Datang di Aplikasi Prediksi Churn!
    
    Aplikasi ini menggunakan model Machine Learning untuk memprediksi 
    apakah pelanggan akan berhenti berlangganan (churn) atau tidak.
    """)
    
    # Metrics
    st.subheader("📊 Performa Model")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "81.2%")
    
    with col2:
        st.metric("Precision", "76.5%")
    
    with col3:
        st.metric("Recall", "54.3%")
    
    with col4:
        st.metric("F1-Score", "63.5%")
    

# ============================================================
# Halaman 2: PREDIKSI
# ============================================================
elif page == "🔮 Prediksi":
    st.title("🔮 Prediksi Churn Pelanggan")
    st.markdown("---")
    
    if model is None:
        st.error("Model tidak ditemukan. Jalankan notebook terlebih dahulu.")
        st.stop()
    
    # Form input
    st.subheader("📝 Masukkan Data Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Memiliki Partner", ["No", "Yes"])
        dependents = st.selectbox("Memiliki Tanggungan", ["No", "Yes"])
        tenure = st.number_input("Lama Berlangganan (bulan)", 0, 100, 12)
        phone_service = st.selectbox("Layanan Telepon", ["No", "Yes"])
    
    with col2:
        internet_service = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 50.0)
        total_charges = st.number_input("Total Biaya ($)", 0.0, 10000.0, 500.0)
    
    # Nilai default untuk fitur lainnya
    default_values = {
        'MultipleLines': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No'
    }
    
    # Tombol prediksi
    if st.button("🎯 Lakukan Prediksi", type="primary", use_container_width=True):
        with st.spinner("Sedang memproses..."):
            # Buat DataFrame
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [default_values['MultipleLines']],
                'InternetService': [internet_service],
                'OnlineSecurity': [default_values['OnlineSecurity']],
                'OnlineBackup': [default_values['OnlineBackup']],
                'DeviceProtection': [default_values['DeviceProtection']],
                'TechSupport': [default_values['TechSupport']],
                'StreamingTV': [default_values['StreamingTV']],
                'StreamingMovies': [default_values['StreamingMovies']],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })
            
            try:
                # Preprocess
                input_processed = preprocessor.transform(input_data)
                
                # Prediksi
                prediction = model.predict(input_processed)[0]
                prediction_proba = model.predict_proba(input_processed)[0]
                
                # Tampilkan hasil
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    if prediction == 1:
                        st.error("🚨 **CHURN**")
                        st.write("Pelanggan berpotensi berhenti berlangganan")
                    else:
                        st.success("✅ **TIDAK CHURN**")
                        st.write("Pelanggan kemungkinan tetap berlangganan")
                
                with col_result2:
                    proba_churn = prediction_proba[1] * 100
                    st.metric("Probabilitas Churn", f"{proba_churn:.1f}%")
                    st.progress(int(proba_churn))
                
                # Detail probabilitas
                st.subheader("📈 Detail Probabilitas")
                proba_df = pd.DataFrame({
                    'Status': ['Tidak Churn', 'Churn'],
                    'Probabilitas': [prediction_proba[0], prediction_proba[1]]
                })
                
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['green', 'red'] if prediction == 1 else ['green', 'lightcoral']
                bars = ax.barh(proba_df['Status'], proba_df['Probabilitas'], color=colors)
                ax.set_xlim(0, 1)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.2%}', ha='left', va='center')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# Halaman 3: ANALISIS
# ============================================================
elif page == "📊 Analisis":
    st.title("📊 Analisis Data")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Distribusi", "Data Sample"])
    
    with tab1:
        st.subheader("Distribusi Churn")
        
        # Data contoh
        churn_data = pd.DataFrame({
            'Status': ['Tidak Churn', 'Churn'],
            'Jumlah': [5174, 1869],
            'Persentase': [73.46, 26.54]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.pie(churn_data['Jumlah'], labels=churn_data['Status'], autopct='%1.1f%%')
            ax1.set_title('Distribusi Churn')
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.bar(churn_data['Status'], churn_data['Jumlah'])
            ax2.set_title('Jumlah Pelanggan')
            st.pyplot(fig2)
    
    with tab2:
        st.subheader("Data Sample")
        try:
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            df_sample = pd.read_csv(url).head(10)
            st.dataframe(df_sample)
        except:
            st.info("Tidak dapat memuat data sample")

# ============================================================
# Halaman 4: INFO MODEL
# ============================================================
else:  # Model Info
    st.title("🤖 Informasi Model")
    st.markdown("---")
    
    st.write("""
    ### Model yang Digunakan
    
    **Random Forest Classifier**
    
    ### Performa Model
    """)
    
    # Tabel metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Nilai': ['81.2%', '76.5%', '54.3%', '63.5%'],
        'Keterangan': [
            'Tingkat akurasi keseluruhan',
            'Akurasi prediksi churn',
            'Kemampuan deteksi churn',
            'Keseimbangan precision & recall'
        ]
    })
    
    st.table(metrics_df)
    
    st.write("""
    ### Fitur Penting
    1. **Contract** - Jenis kontrak
    2. **Tenure** - Lama berlangganan  
    3. **InternetService** - Jenis internet
    4. **MonthlyCharges** - Biaya bulanan
    5. **PaymentMethod** - Cara pembayaran
    """)

# Footer
st.markdown("---")
st.markdown("**UAS Bengkel Koding Data Science - 2024**")