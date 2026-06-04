import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os
import hasil_penelitian
import informasi

# --- KONFIGURASI ---
st.set_page_config(page_title="Leaf Disease Analyzer", page_icon="🌿", layout="wide")
REPO_OWNER = "blasterdark300"
REPO_NAME = "leaf-disease-comparison-model"
MODELS_BASE_PATH = "output/models" 
LABELS_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/output/labels/labels.json"

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
    
    # Input Gambar
    image_file = st.file_uploader("Pilih gambar dari perangkat Anda", type=["jpg", "png", "jpeg"])
    
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Gambar yang dianalisis", width=300)
        
        if st.button("Analisis 3 Model"):
            class_names = requests.get(LABELS_URL).json()
            cols = st.columns(3)
            batch_models = {k: v for k, v in model_dict.items() if k.startswith(selected_batch)}
            
            for i, (key, path) in enumerate(batch_models.items()):
                with cols[i]:
                    model = load_model(path)
                    img_proc = image.convert('RGB').resize((256, 256))
                    img_array = np.expand_dims(np.array(img_proc) / 255.0, axis=0)
                    preds = model.predict(img_array)
                    label = class_names[np.argmax(preds)]
                    
                    st.write(f"Model: {key.split('/')[1]}")
                    st.success(f"Hasil: **{label}**")
                    
                    val = st.radio(f"Validasi {key}", ["Belum", "Benar", "Salah"], key=f"val_{key}")
                    if st.button(f"Simpan {key.split('/')[1]}", key=f"btn_{key}"):
                        if 'history' not in st.session_state: st.session_state.history = []
                        st.session_state.history.append({"Model": key, "Hasil": label, "Validasi": val})
                        st.success("Tersimpan!")

# --- HALAMAN LAINNYA ---
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
        st.download_button("📥 Unduh CSV", csv, "histori_validasi.csv", "text/csv")
    else:
        st.write("Belum ada data validasi.")
