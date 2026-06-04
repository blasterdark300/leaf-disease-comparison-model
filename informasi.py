import streamlit as st

def render():
    st.title("ℹ️ Informasi Proyek")
    
    st.markdown("""
    ### Tentang Aplikasi
    Aplikasi **Leaf Disease Analyzer** ini merupakan sistem berbasis Deep Learning yang dirancang untuk mendeteksi penyakit pada berbagai jenis tanaman secara otomatis.
    
    ### Arsitektur Model yang Digunakan
    Sistem ini membandingkan kinerja tiga arsitektur model utama:
    1. **Custom CNN**: Arsitektur model yang dibangun dari awal (*scratch*) untuk mempelajari pola fitur daun.
    2. **MobileNetV2**: Model *pre-trained* yang dioptimalkan untuk kecepatan dan efisiensi pada perangkat *mobile* atau *edge*.
    3. **InceptionV3**: Model *pre-trained* dengan arsitektur kompleks yang dirancang untuk menangkap fitur dalam berbagai skala resolusi.
    
    ### Metodologi
    * **Batch Size**: Eksperimen dilakukan dengan variasi *batch size* (8, 16, dan 32) untuk melihat pengaruhnya terhadap konvergensi model dan akurasi validasi.
    * **Preprocessing**: Gambar di-*resize* ke ukuran 256x256 piksel dengan normalisasi nilai piksel ke rentang [0, 1].
    * **Validasi**: Sistem memungkinkan pengguna untuk memvalidasi hasil prediksi model secara langsung, yang kemudian disimpan untuk keperluan evaluasi lebih lanjut.
    
    ---
    *Proyek ini dikembangkan sebagai bagian dari riset komparatif performa model deep learning pada klasifikasi penyakit tanaman.*
    """)
