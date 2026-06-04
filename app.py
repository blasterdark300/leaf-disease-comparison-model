import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os
import hasil_penelitian
import informasi  # Pastikan Anda sudah membuat file informasi.py

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
    
    # Inisialisasi state gambar
    if 'img_to_analyze' not in st.session_state: st.session_state.img_to_analyze = None
    
    tab1, tab2 = st.tabs(["📸 Kamera", "📂 Galeri"])
    with tab1:
        camera_file = st.camera_input("Ambil foto")
        if camera_file: st.session_state.img_to_analyze = Image.open(camera_file)
    with tab2:
        gallery = requests.get(GALLERY_URL).json()
        kat = st.selectbox("Pilih Kategori:", list(gallery.keys()))
        col1, col2 = st.columns(2)
        if col1.button("Muat Gambar 1"): st.session_state.img_to_analyze = Image.open(requests.get(gallery[kat][0], stream=True).raw)
        if col2.button("Muat Gambar 2"): st.session_state.img_to_analyze = Image.open(requests.get(gallery[kat][1], stream=True).raw)
    
    if st.session_state.img_to_analyze:
        st.image(st.session_state.img_to_analyze, caption="Gambar yang dianalisis", width=300)
        
        if st.button("Analisis 3 Model"):
            class_names = requests.get(LABELS_URL).json()
            cols = st.columns(3)
            batch_models = {k:v for k,v in model_dict.items() if k.startswith(selected_batch)}
            
            for i, (key, path) in enumerate(batch_models.items()):
                with cols[i]:
                    model = load_model(path)
                    img_proc = st.session_state.img_to_analyze.convert('RGB').resize((256, 256))
                    img_array = np.expand_dims(np.array(img_proc) / 255.0, axis=0)
                    pred = model.predict(img_array)
                    label = class_names[np.argmax(pred)]
                    
                    st.write(f"Model: {key.split('/')[1]}")
                    st.success(f"Hasil: **{label}**")
                    
                    val = st.radio(f"Validasi", ["Belum", "Benar", "Salah"], key=f"val_{key}")
                    if st.button(f"Simpan {key.split('/')[1]}", key=f"btn_{key}"):
                        if 'history' not in st.session_state: st.session_state.history = []
                        st.session_state.history.append({"Model": key, "Hasil": label, "Validasi": val})
                        st.balloons()

elif menu == "Hasil Penelitian":
    hasil_penelitian.render()
elif menu == "Informasi":
    informasi.render()
elif menu == "Histori":
    st.title("🕒 Histori Validasi")
    if 'history' in st.session_state and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh CSV", csv, "histori.csv", "text/csv")
