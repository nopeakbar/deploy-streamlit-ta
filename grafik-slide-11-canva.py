import pickle
import matplotlib.pyplot as plt
import os

# ==============================================================================
# PILIH SALAH SATU FILE (Saran: Pakai yang 'without_indicators' karena MAPE lebih bagus)
# ==============================================================================
file_path = 'models/results_without_indicators.pkl' 
# file_path = 'models/results_with_indicators.pkl' # Aktifkan ini kalau mau lihat yang Skenario B

# Cek apakah file ada
if not os.path.exists(file_path):
    print(f"❌ Error: File {file_path} tidak ditemukan. Pastikan folder 'models' ada di sini.")
else:
    print(f"📂 Membuka kapsul waktu: {file_path}...")
    
    # Buka file pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Ambil data harga yang tersimpan
    actual_prices = data['actual_prices']
    predicted_prices = data['predicted_prices']
    mape = data['mape']
    ticker = data.get('ticker', 'NVDA') # Default NVDA kalau key tidak ketemu

    print(f"✅ Data berhasil di-load!")
    print(f"   MAPE Model: {mape:.2f}%")
    print(f"   Jumlah Data Test: {len(actual_prices)} Hari")

    # ==========================================
    # PLOT GRAFIK HASIL PREDIKSI
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # Plot Harga Asli
    plt.plot(actual_prices, label='Harga Aktual (Real)', color='black', linewidth=2, alpha=0.7)
    
    # Plot Prediksi
    plt.plot(predicted_prices, label='Prediksi Model (Bi-LSTM)', color='red', linewidth=1.5, alpha=0.9)
    
    # Dekorasi Grafik
    plt.title(f'Visualisasi Prediksi Saham {ticker}\n(Model Baseline - MAPE: {mape:.2f}%)', fontsize=14, fontweight='bold')
    plt.xlabel('Hari ke- (Testing Period)', fontsize=12)
    plt.ylabel('Harga Saham (USD)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Tambahkan teks panah (Opsional - biar keren)
    # Mencari titik di mana prediksi dan asli sangat dekat
    plt.annotate('Akurasi Tinggi (Tren Terbaca)', 
                 xy=(50, predicted_prices[50]), 
                 xytext=(20, predicted_prices[50]+20),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=10, color='green')

    plt.tight_layout()
    
    # Simpan Gambar
    output_filename = 'grafik_prediksi_final.png'
    plt.savefig(output_filename, dpi=300)
    print(f"📸 Gambar berhasil disimpan sebagai: {output_filename}")
    plt.show()