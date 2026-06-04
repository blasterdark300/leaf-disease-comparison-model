import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os

# ... (Konfigurasi dan Fungsi get_model_list & load_model tetap sama) ...

# --- HALAMAN DETEKSI ---
if menu == "Deteksi Penyakit":
    st.title("🌿 Comparative Leaf Disease Analyzer")
    
    # 1. Menu Pilihan Batch Size
    model_dict = get_model_list() # Mengembalikan format: {'8/model.h5': 'path', '16/model.h5': 'path'}
    
    # Ambil list batch yang unik (8, 16, 32)
    available_batches = sorted(list(set([k.split('/')[0] for k in model_dict.keys()])))
    selected_batch = st.selectbox("Pilih Batch Size:", available_batches)
    
    # 2. Input Gambar
    image = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])
    
    if image and st.button("Analisis 3 Model"):
        img = Image.open(image)
        st.image(img, caption="Gambar yang dianalisis", width=300)
        
        class_names = requests.get(LABELS_URL).json()
        cols = st.columns(3)
        
        # Filter model berdasarkan batch yang dipilih
        batch_models = {k: v for k, v in model_dict.items() if k.startswith(selected_batch)}
        
        if 'history' not in st.session_state: st.session_state.history = []

        for i, (key, path) in enumerate(batch_models.items()):
            with cols[i]:
                st.info(f"Model: {key.split('/')[1]}")
                model = load_model(path)
                
                # Prediksi
                img_proc = img.convert('RGB').resize((256, 256))
                img_array = np.expand_dims(np.array(img_proc) / 255.0, axis=0)
                preds = model.predict(img_array)
                label = class_names[np.argmax(preds)]
                
                st.write(f"Hasil: **{label}**")
                
                # Validasi unik per model
                val = st.radio(f"Validasi {key}", ["Belum", "Benar", "Salah"], key=f"val_{key}")
                
                if st.button(f"Simpan {key.split('/')[1]}", key=f"btn_{key}"):
                    st.session_state.history.append({"Model": key, "Hasil": label, "Validasi": val})
                    st.success("Tersimpan!")
