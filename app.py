import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os

# --- KONFIGURASI ---
st.set_page_config(page_title="Leaf Disease Analyzer", page_icon="🌿", layout="centered")

REPO_OWNER = "blasterdark300"
REPO_NAME = "leaf-disease-comparison-model"
MODELS_PATH = "output/models"
LABELS_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/output/labels/labels.json"

# --- FUNGSI ---
@st.cache_data
def get_model_list():
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{MODELS_PATH}"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        return [f['name'] for f in files if f['name'].endswith(('.h5', '.keras'))]
    return []

@st.cache_resource
def load_model(filename):
    url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/main/{MODELS_PATH}/{filename}"
    local_path = filename
    if not os.path.exists(local_path):
        response = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(local_path, compile=False)

# --- NAVIGASI SIDEBAR ---
menu = st.sidebar.radio("Menu", ["Deteksi Penyakit", "Hasil Penelitian", "Informasi", "Histori"])

# --- HALAMAN DETEKSI ---
if menu == "Deteksi Penyakit":
    st.title("🌿 Leaf Disease Analyzer")
    model_files = get_model_list()
    if not model_files: st.error("Model tidak ditemukan."); st.stop()
    
    selected_model = st.selectbox("Pilih Model:", model_files)
    model = load_model(selected_model)
    class_names = requests.get(LABELS_URL).json()

    tab1, tab2 = st.tabs(["📸 Kamera", "📂 Galeri"])
    image = None
    with tab1:
        camera_file = st.camera_input("Ambil foto")
        if camera_file: image = Image.open(camera_file)
    with tab2:
        uploaded = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])
        if uploaded: image = Image.open(uploaded)

    if image:
        st.image(image, caption="Gambar yang dianalisis")
        if st.button("Analisis"):
            img = image.convert('RGB').resize((256, 256))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = model.predict(img_array)
            label = class_names[np.argmax(preds)]
            st.success(f"Hasil Prediksi: **{label}**")
            
            if 'history' not in st.session_state: st.session_state.history = []
            st.session_state.history.append({"Model": selected_model, "Hasil": label})

# --- HALAMAN HASIL PENELITIAN ---
elif menu == "Hasil Penelitian":
    st.title("📊 Hasil Perbandingan Model")
    st.markdown("""
    | Model | Akurasi | Efisiensi |
    | :--- | :--- | :--- |
    | Custom CNN | 92% | Sedang |
    | MobileNetV2 | 95% | Tinggi |
    | InceptionV3 | 98% | Rendah |
    """)

# --- HALAMAN INFORMASI ---
elif menu == "Informasi":
    st.title("ℹ️ Informasi")
    st.write("Aplikasi ini menggunakan deep learning untuk deteksi penyakit daun.")

# --- HALAMAN HISTORI ---
elif menu == "Histori":
    st.title("🕒 Histori Prediksi")
    if 'history' in st.session_state and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh CSV", csv, "histori_prediksi.csv", "text/csv")
    else:
        st.write("Belum ada histori deteksi.")
