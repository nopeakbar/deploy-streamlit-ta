import pickle
import tensorflow as tf
import os

print("🔍 Testing Load Files...")

# Cek Path
base_path = 'models/'

try:
    # 1. Test Load Model
    model = tf.keras.models.load_model(os.path.join(base_path, 'lstm_without_indicators.keras'))
    print("✅ Model .keras loaded successfully")

    # 2. Test Load Pickle Results
    with open(os.path.join(base_path, 'results_without_indicators.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Pickle loaded. Keys found: {data.keys()}")
    print(f"   RMSE tersimpan: {data['rmse']}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")