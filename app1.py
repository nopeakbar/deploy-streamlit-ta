import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image # Import untuk memproses gambar lokal

# ==========================================
# 0. KONFIGURASI GAMBAR USER (GANTI DISINI)
# ==========================================
# Ganti 'fotoku.jpg' dengan nama file fotomu yang ada di folder yang sama
FOTO_PATH = 'nopeakbar.png' 

# Load Gambar untuk Ikon Tab & Sidebar
try:
    user_image = Image.open(FOTO_PATH)
    page_icon_img = user_image
except FileNotFoundError:
    # Fallback jika file tidak ditemukan
    user_image = None
    page_icon_img = "🎓"

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Dashboard Skripsi - LSTM NVIDIA",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=page_icon_img # Menggunakan foto lokal sebagai ikon tab
)

# ==========================================
# 2. CSS LIGHT MODE (High Contrast + Round Image)
# ==========================================
st.markdown("""
<style>
    /* Metric Box */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #444444 !important;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* Bikin Foto di Sidebar jadi Bulat */
    img[data-testid="stImage"] {
        border-radius: 50%;
        object-fit: cover;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spacer atas */
    .block-container { padding-top: 2rem; }
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
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    with st.container():
        # Menampilkan Foto Lokal
        if user_image is not None:
            st.image(user_image, width=120) # Atur ukuran foto disini
        else:
            # Gambar default jika foto lokal gagal dimuat
            st.image("https://cdn-icons-png.flaticon.com/512/3426/3426653.png", width=70)
            if FOTO_PATH != 'fotoku.jpg': # Hanya warn jika user sudah mencoba ganti path
                st.warning(f"File '{FOTO_PATH}' tidak ditemukan.")

        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin-top: 10px;'>
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
    st.subheader("🎛️ Kontrol Garis")
    show_actual = st.checkbox("⚫ Harga Asli (Actual)", value=True)
    show_baseline = st.checkbox("🔴 Model Baseline (No Ind)", value=True)
    show_proposed = st.checkbox("🟢 Model Proposed (With Ind)", value=True)

# ==========================================
# 5. JUDUL & DATA PREP
# ==========================================
st.markdown("""
<div style='background-color: #f8f9fa; padding: 20px; border-left: 6px solid #1f77b4; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 25px;'>
    <strong style='color: #1f77b4; font-size: 14px; text-transform: uppercase;'>Judul Tugas Akhir:</strong>
    <h3 style='color: #212529; margin-top: 5px; margin-bottom: 0; font-size: 20px;'>
        Analisis Pengaruh Fitur Teknikal Terhadap Akurasi Prediksi Harga Saham NVIDIA Menggunakan Algoritma LSTM
    </h3>
</div>
""", unsafe_allow_html=True)

# --- LOGIKA DATA ---
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

# Gabungkan ke DataFrame Master
df_master = pd.DataFrame(index=dates)
if show_actual:
    df_master['Actual Price'] = plot_actual
if show_baseline:
    df_master['Baseline (No Indicators)'] = plot_baseline
if show_proposed and (plot_proposed is not None):
    df_master['Proposed (With Indicators)'] = plot_proposed

# ==========================================
# 6. FITUR ZOOM PINTAR (SLIDER)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Data", "📋 Data Mentah", "📝 Kesimpulan"])

with tab1:
    st.markdown("#### 🔎 Filter Rentang Waktu (Zoom)")
    
    start_date, end_date = st.select_slider(
        "Geser titik kiri dan kanan untuk Zoom In/Out:",
        options=dates,
        value=(dates[0], dates[-1]),
        format_func=lambda x: x.strftime("%d %b %Y")
    )
    
    # Filter Data Berdasarkan Slider
    df_filtered = df_master.loc[start_date:end_date]
    
    # Tampilkan Grafik Native (Super Smooth)
    colors = []
    if show_actual: colors.append("#000000")
    if show_baseline: colors.append("#FF4B4B")
    if show_proposed: colors.append("#008000")
    
    if not df_filtered.empty:
        st.line_chart(
            df_filtered, 
            color=colors if colors else None,
            use_container_width=True
        )
        st.caption(f"Menampilkan data dari **{start_date.strftime('%d %b %Y')}** sampai **{end_date.strftime('%d %b %Y')}**")
    else:
        st.warning("Silakan pilih minimal satu model di sidebar.")

    # Evaluasi Metrik
    st.markdown("#### 🧮 Evaluasi Performa")
    if results['proposed']:
        col1, col2, col3 = st.columns(3)
        rmse_base, rmse_prop = results['baseline']['rmse'], results['proposed']['rmse']
        mape_base, mape_prop = results['baseline']['mape'], results['proposed']['mape']
        acc_base, acc_prop = results['baseline']['accuracy']*100, results['proposed']['accuracy']*100
        
        col1.metric("RMSE (Error)", f"{rmse_prop:.4f}", f"{rmse_base - rmse_prop:.4f}", delta_color="inverse")
        col2.metric("MAPE (Error)", f"{mape_prop:.2f}%", f"{mape_base - mape_prop:.2f}%", delta_color="inverse")
        col3.metric("Directional Accuracy", f"{acc_prop:.2f}%", f"{acc_prop - acc_base:.2f}%", delta_color="normal")
    else:
        st.metric("Baseline RMSE", f"{results['baseline']['rmse']:.4f}")

with tab2:
    st.subheader("🔍 Detail Angka (Sesuai Zoom)")
    st.dataframe(df_filtered.style.format("{:.2f}"), use_container_width=True)
    
    csv = df_filtered.to_csv().encode('utf-8')
    st.download_button("📥 Download Data Terfilter", csv, "hasil_filter.csv", "text/csv")

with tab3:
    st.subheader("📝 Kesimpulan Sementara")
    st.info("ℹ️ **Ringkasan:** Hasil eksperimen menunjukkan dampak indikator teknikal pada akurasi prediksi.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Baseline RMSE:** {results['baseline']['rmse']:.4f}")
    with col_b:
        val = results['proposed']['rmse'] if results['proposed'] else 0
        st.write(f"**Proposed RMSE:** {val:.4f}")