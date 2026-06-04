import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os
import hasil_penelitian # Memanggil file modular

# --- KONFIGURASI ---
st.set_page_config(page_title="Leaf Disease Analyzer", page_icon="🌿", layout="wide")
REPO_OWNER = "blasterdark300"
REPO_NAME = "leaf-disease-comparison-model"
MODELS_BASE_PATH = "output/models" 
LABELS_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/output/labels/labels.json"
GALLERY_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/data_galeri.json"

# --- FUNGSI ---
@st.cache_data
def get_model_list():
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{MODELS_BASE_PATH}"
    response = requests.get(api_url)
    model_options = {}
    if response.status_code == 200:
        folders = [f['name'] for f in response.json() if f['type'] == 'dir']
        for folder in folders:
            files_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{MODELS_BASE_PATH}/{folder}"
            files_resp = requests.get(files_url)
            if files_resp.status_code == 200:
                for f in files_resp.json():
                    if f['name'].endswith('.h5'):
                        key = f"{folder}/{f['name']}"
                        model_options[key] = f"{MODELS_BASE_PATH}/{key}"
    return model_options

@st.cache_resource
def load_model(path_in_repo):
    url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/main/{path_in_repo}"
    local_filename = path_in_repo.replace("/", "_")
    if not os.path.exists(local_filename):
        response = requests.get(url)
        with open(local_filename, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(local_filename, compile=False)

# --- SIDEBAR ---
menu = st.sidebar.radio("Menu", ["Deteksi Penyakit", "Hasil Penelitian", "Informasi", "Histori"])

# --- HALAMAN DETEKSI ---
if menu == "Deteksi Penyakit":
    st.title("🌿 Comparative Leaf Disease Analyzer")
    model_dict = get_model_list()
    available_batches = sorted(list(set([k.split('/')[0] for k in model_dict.keys()])))
    selected_batch = st.selectbox("Pilih Batch Size:", available_batches)
    
    tab1, tab2 = st.tabs(["📸 Kamera", "📂 Galeri"])
    image = None
    with tab1:
        camera_file = st.camera_input("Ambil foto")
        if camera_file: image = Image.open(camera_file)
    with tab2:
        gallery = requests.get(GALLERY_URL).json()
        kat = st.selectbox("Pilih Kategori:", list(gallery.keys()))
        col1, col2 = st.columns(2)
        if col1.button("Muat Gambar 1"): image = Image.open(requests.get(gallery[kat][0], stream=True).raw)
        if col2.button("Muat Gambar 2"): image = Image.open(requests.get(gallery[kat][1], stream=True).raw)
    
    if image and st.button("Analisis 3 Model"):
        st.image(image, width=300)
        class_names = requests.get(LABELS_URL).json()
        cols = st.columns(3)
        for i, (key, path) in enumerate({k:v for k,v in model_dict.items() if k.startswith(selected_batch)}.items()):
            with cols[i]:
                model = load_model(path)
                preds = model.predict(np.expand_dims(np.array(image.convert('RGB').resize((256, 256)))/255.0, axis=0))
                st.write(f"Model {key.split('/')[1]}: **{class_names[np.argmax(preds)]}**")
                val = st.radio(f"Validasi {key}", ["Belum", "Benar", "Salah"], key=f"val_{key}")
                if st.button(f"Simpan {key.split('/')[1]}", key=f"btn_{key}"):
                    if 'history' not in st.session_state: st.session_state.history = []
                    st.session_state.history.append({"Model": key, "Hasil": class_names[np.argmax(preds)], "Validasi": val})

# --- ROUTING LAINNYA ---
elif menu == "Hasil Penelitian":
    hasil_penelitian.render()
elif menu == "Informasi":
    st.title("ℹ️ Informasi")
    st.write("Aplikasi deteksi penyakit menggunakan model deep learning (CNN, MobileNetV2, InceptionV3).")
elif menu == "Histori":
    if 'history' in st.session_state: st.table(pd.DataFrame(st.session_state.history))
