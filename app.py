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
# Path dasar di GitHub
MODELS_BASE_PATH = "output/models" 
LABELS_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/output/labels/labels.json"

# --- FUNGSI ---
@st.cache_data
def get_model_list():
    """Mengambil daftar model dari folder 8, 16, 32 di GitHub"""
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{MODELS_BASE_PATH}"
    response = requests.get(api_url)
    model_options = {}
    
    if response.status_code == 200:
        # Mengambil daftar folder (8, 16, 32)
        folders = [f['name'] for f in response.json() if f['type'] == 'dir']
        for folder in folders:
            files_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{MODELS_BASE_PATH}/{folder}"
            files_resp = requests.get(files_url)
            if files_resp.status_code == 200:
                for f in files_resp.json():
                    if f['name'].endswith('.h5'):
                        # Key untuk tampilan, Value untuk path download
                        key = f"{folder}/{f['name']}"
                        model_options[key] = f"{MODELS_BASE_PATH}/{key}"
    return model_options

@st.cache_resource
def load_model(path_in_repo):
    """Mendownload dan memuat model dari GitHub"""
    url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/main/{path_in_repo}"
    # Ubah slash menjadi underscore untuk nama file lokal agar aman
    local_filename = path_in_repo.replace("/", "_") 
    
    if not os.path.exists(local_filename):
        response = requests.get(url)
        with open(local_filename, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(local_filename, compile=False)

# --- NAVIGASI SIDEBAR ---
menu = st.sidebar.radio("Menu", ["Deteksi Penyakit", "Hasil Penelitian", "Informasi", "Histori"])

# --- HALAMAN DETEKSI ---
if menu == "Deteksi Penyakit":
    st.title("🌿 Leaf Disease Analyzer")
    model_dict = get_model_list()
    if not model_dict: 
        st.error("Model tidak ditemukan. Pastikan folder dan file sudah di-push ke GitHub."); st.stop()
    
    selected_key = st.selectbox("Pilih Model (Batch Size/Nama):", list(model_dict.keys()))
    model = load_model(model_dict[selected_key])
    
    # Load Labels
    try:
        class_names = requests.get(LABELS_URL).json()
    except:
        st.error("Gagal memuat labels.json"); st.stop()

    tab1, tab2 = st.tabs(["📸 Kamera", "📂 Galeri"])
    image = None
    with tab1:
        camera_file = st.camera_input("Ambil foto")
        if camera_file: image = Image.open(camera_file)
    with tab2:
        uploaded = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])
        if uploaded: image = Image.open(uploaded)

    if image:
        st.image(image, caption="Gambar yang dianalisis", use_container_width=True)
        if st.button("Analisis"):
            with st.spinner("Sedang memproses..."):
                img = image.convert('RGB').resize((256, 256))
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                preds = model.predict(img_array)
                label = class_names[np.argmax(preds)]
                st.success(f"Hasil Prediksi: **{label}**")
                
                if 'history' not in st.session_state: st.session_state.history = []
                st.session_state.history.append({"Model": selected_key, "Hasil": label})

# --- HALAMAN LAINNYA ---
elif menu == "Hasil Penelitian":
    st.title("📊 Hasil Perbandingan Model")
    st.markdown("""
    | Model | Akurasi | Efisiensi |
    | :--- | :--- | :--- |
    | Custom CNN | 92% | Sedang |
    | MobileNetV2 | 95% | Tinggi |
    | InceptionV3 | 98% | Rendah |
    """)

elif menu == "Informasi":
    st.title("ℹ️ Informasi")
    st.write("Aplikasi ini menggunakan deep learning untuk deteksi penyakit daun berdasarkan konfigurasi batch size yang berbeda.")

elif menu == "Histori":
    st.title("🕒 Histori Prediksi")
    if 'history' in st.session_state and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh CSV", csv, "histori_prediksi.csv", "text/csv")
    else:
        st.write("Belum ada histori deteksi.")
