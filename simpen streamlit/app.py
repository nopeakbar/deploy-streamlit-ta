import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Dashboard Skripsi - LSTM NVIDIA",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

# ==========================================
# 2. CSS KHUSUS UNTUK HIGH CONTRAST (LIGHT MODE)
# ==========================================
# Kita paksa warna teks jadi gelap agar kontras dengan background putih
st.markdown("""
<style>
    /* 1. Styling untuk Metric Box (Kotak Angka Error) */
    div[data-testid="stMetric"] {
        background-color: #ffffff; /* Background Putih */
        border: 1px solid #e0e0e0; /* Border Abu tipis */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); /* Bayangan halus */
    }
    
    /* Warna Label (RMSE, MAPE) */
    div[data-testid="stMetricLabel"] {
        color: #444444 !important; /* Abu tua */
        font-weight: bold;
    }
    
    /* Warna Angka (Value) */
    div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Hitam Pekat */
    }
    
    /* 2. Hapus elemen pengganggu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 3. Styling Tab agar lebih jelas */
    button[data-baseweb="tab"] {
        font-weight: bold;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI LOAD DATA
# ==========================================
@st.cache_resource
def load_results():
    path = 'models/' 
    data = {}
    
    if not os.path.exists(path):
        if os.path.exists('../models/'):
            path = '../models/'
        else:
            st.error("❌ Folder 'models' tidak ditemukan!")
            return None

    try:
        with open(os.path.join(path, 'results_without_indicators.pkl'), 'rb') as f:
            data['baseline'] = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ File 'results_without_indicators.pkl' hilang.")
        return None

    try:
        with open(os.path.join(path, 'results_with_indicators.pkl'), 'rb') as f:
            data['proposed'] = pickle.load(f)
    except FileNotFoundError:
        data['proposed'] = None
        
    return data

results = load_results()
if results is None: st.stop()

# ==========================================
# 4. SIDEBAR (PROFILE)
# ==========================================
with st.sidebar:
    # Menggunakan container agar lebih rapi
    with st.container():
        st.image("https://cdn-icons-png.flaticon.com/512/3426/3426653.png", width=70)
        st.markdown("### Profil Mahasiswa")
        
        # Menggunakan st.markdown dengan HTML inline untuk kontrol warna penuh
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd;'>
            <p style='margin:0; color:#555; font-size:12px;'>Nama Mahasiswa</p>
            <p style='margin:0; color:#000; font-weight:bold; font-size:16px;'>Noveanto Nur Akbar</p>
            <br>
            <p style='margin:0; color:#555; font-size:12px;'>NIM</p>
            <p style='margin:0; color:#000; font-weight:bold; font-size:16px;'>123220129</p>
            <br>
            <p style='margin:0; color:#555; font-size:12px;'>Program Studi</p>
            <p style='margin:0; color:#000; font-weight:bold;'>Informatika</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("🎛️ Kontrol Grafik")
    show_actual = st.checkbox("✅ Harga Asli (Actual)", value=True)
    show_baseline = st.checkbox("🔴 Model Baseline (No Ind)", value=True)
    show_proposed = st.checkbox("🟢 Model Proposed (With Ind)", value=True)
    
    st.markdown("---")
    st.info("💡 **Tips:** Hover mouse di atas grafik untuk melihat angka detailnya.")

# ==========================================
# 5. HEADER & JUDUL (FIX WARNA GELAP)
# ==========================================
st.title("📈 Analisis Prediksi Saham NVIDIA (LSTM)")

# --- PERBAIKAN DI SINI ---
# Menggunakan background abu muda (#f8f9fa) dan teks hitam (#212529)
# Tidak ada lagi blok hitam yang bikin pusing.
st.markdown("""
<div style='
    background-color: #f8f9fa; 
    padding: 20px; 
    border-left: 6px solid #1f77b4; 
    border-radius: 8px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 25px;
'>
    <strong style='color: #1f77b4; font-size: 14px; text-transform: uppercase;'>Judul Tugas Akhir:</strong>
    <h3 style='color: #212529; margin-top: 5px; margin-bottom: 0; font-size: 20px;'>
        Analisis Pengaruh Fitur Teknikal Terhadap Akurasi Prediksi Harga Saham NVIDIA Menggunakan Algoritma LSTM
    </h3>
</div>
""", unsafe_allow_html=True)

# Data Processing (Slicing Logic)
data_actual = results['baseline']['actual_prices'].flatten()
data_baseline_pred = results['baseline']['predicted_prices'].flatten()

if results['proposed']:
    data_proposed_pred = results['proposed']['predicted_prices'].flatten()
    min_len = min(len(data_baseline_pred), len(data_proposed_pred), len(data_actual))
    
    plot_actual = data_actual[-min_len:]
    plot_baseline = data_baseline_pred[-min_len:]
    plot_proposed = data_proposed_pred[-min_len:]
else:
    plot_actual = data_actual
    plot_baseline = data_baseline_pred
    plot_proposed = None
    min_len = len(plot_actual)

dates = pd.date_range(end='2024-12-31', periods=min_len, freq='B')

# ==========================================
# 6. VISUALISASI UTAMA
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Data", "📋 Data Mentah", "📝 Kesimpulan Analisis"])

with tab1:
    st.markdown(f"#### Perbandingan Pergerakan Harga ({min_len} Hari Terakhir)")
    
    chart_data = pd.DataFrame(index=dates)
    
    if show_actual:
        chart_data['Actual Price'] = plot_actual
    if show_baseline:
        chart_data['Baseline (No Indicators)'] = plot_baseline
    if show_proposed and (plot_proposed is not None):
        chart_data['Proposed (With Indicators)'] = plot_proposed
    
    if not chart_data.empty:
        # PENTING: Warna diset kontras untuk Light Mode
        # Hitam (Actual), Merah Terang (Baseline), Hijau Gelap (Proposed)
        st.line_chart(
            chart_data, 
            color=["#000000", "#FF4B4B", "#008000"] if (show_actual and show_baseline and show_proposed) else None
        )
    
    st.markdown("#### 🧮 Evaluasi Performa Model")
    if results['proposed']:
        col1, col2, col3 = st.columns(3)
        
        rmse_base, rmse_prop = results['baseline']['rmse'], results['proposed']['rmse']
        mape_base, mape_prop = results['baseline']['mape'], results['proposed']['mape']
        acc_base, acc_prop = results['baseline']['accuracy']*100, results['proposed']['accuracy']*100
        
        # Metric dengan Logic Warna Inverse (Error Turun = Bagus = Hijau)
        col1.metric("RMSE (Error)", f"{rmse_prop:.4f}", f"{rmse_base - rmse_prop:.4f}", delta_color="inverse")
        col2.metric("MAPE (Error)", f"{mape_prop:.2f}%", f"{mape_base - mape_prop:.2f}%", delta_color="inverse")
        col3.metric("Directional Accuracy", f"{acc_prop:.2f}%", f"{acc_prop - acc_base:.2f}%", delta_color="normal")
    else:
        st.metric("Baseline RMSE", f"{results['baseline']['rmse']:.4f}")

with tab2:
    st.subheader("🔍 Detail Angka Prediksi")
    df_raw = pd.DataFrame({'Actual': plot_actual, 'Pred_Baseline': plot_baseline}, index=dates)
    if plot_proposed is not None:
        df_raw['Pred_Proposed'] = plot_proposed
        
    st.dataframe(df_raw.style.format("{:.2f}"), use_container_width=True)
    csv = df_raw.to_csv().encode('utf-8')
    st.download_button("📥 Download Excel/CSV", csv, "hasil_prediksi.csv", "text/csv")

with tab3:
    st.subheader("📝 Kesimpulan Sementara")
    
    # Menggunakan st.success/warning agar warnanya otomatis pas dengan tema
    st.info("ℹ️ **Ringkasan:** Hasil eksperimen menunjukkan dampak indikator teknikal pada akurasi prediksi.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### ✅ Model Baseline")
        st.write(f"- **RMSE:** {results['baseline']['rmse']:.4f}")
        st.write("- **Karakteristik:** Lebih stabil mengikuti tren harga.")
        
    with col_b:
        st.markdown("##### ⚠️ Model Proposed")
        val_rmse = results['proposed']['rmse'] if results['proposed'] else 0
        st.write(f"- **RMSE:** {val_rmse:.4f}")
        st.write("- **Karakteristik:** Lebih sensitif terhadap volatilitas, namun perlu tuning lebih lanjut.")