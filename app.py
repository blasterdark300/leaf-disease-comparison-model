import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import json
import os

# --- KONFIGURASI ---
st.set_page_config(page_title="Leaf Disease Analyzer", page_icon="🌿", layout="centered")

REPO_OWNER = "blasterdark300"
REPO_NAME = "leaf-disease-comparison-model"
MODELS_PATH = "output/models"
LABELS_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/output/labels/labels.json"

# --- FUNGSI MENGAMBIL DAFTAR MODEL DARI GITHUB ---
@st.cache_data
def get_model_list():
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{MODELS_PATH}"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        # Filter hanya file .h5 dan .keras
        return [f['name'] for f in files if f['name'].endswith(('.h5', '.keras'))]
    return []

# --- FUNGSI MEMUAT MODEL ---
@st.cache_resource
def load_model(filename):
    url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/main/{MODELS_PATH}/{filename}"
    local_path = filename
    if not os.path.exists(local_path):
        response = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(local_path)

@st.cache_data
def get_labels():
    return requests.get(LABELS_URL).json()

# --- UI APLIKASI ---
st.title("🌿 Leaf Disease Analyzer")
model_files = get_model_list()

if not model_files:
    st.error("Tidak ada model ditemukan di folder.")
    st.stop()

selected_model = st.selectbox("Pilih Model:", model_files)
model = load_model(selected_model)
class_names = get_labels()

# --- INPUT & PREDIKSI ---
tab1, tab2 = st.tabs(["📸 Kamera", "📂 Galeri"])
image = None

with tab1:
    camera_file = st.camera_input("Ambil foto")
    if camera_file: image = Image.open(camera_file)

with tab2:
    uploaded = st.file_uploader("Pilih gambar", type=["jpg", "png"])
    if uploaded: image = Image.open(uploaded)

if image:
    st.image(image, use_container_width=True)
    if st.button("Analisis"):
        # Preprocessing 256x256
        img = image.convert('RGB').resize((256, 256))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        preds = model.predict(img_array)
        label = class_names[np.argmax(preds)]
        st.success(f"Prediksi: **{label}**")
