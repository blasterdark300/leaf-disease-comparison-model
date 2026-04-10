# 🍃 Leaf Disease Comparison Model
### A Comparative Analysis of Custom CNN, MobileNetV2, and InceptionV3 for Multi-Plant Leaf Disease Identification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Framework](https://img.shields.io/badge/Framework-Keras-red.svg)
![GitHub License](https://img.shields.io/github/license/blasterdark300/leaf-disease-comparison-model)

## 📌 Project Overview
Proyek ini merupakan penelitian komparatif untuk mengidentifikasi berbagai jenis penyakit tanaman berdasarkan citra daun. Mengingat sektor pertanian adalah pilar ekonomi yang vital, deteksi penyakit secara dini sangat krusial. Penelitian ini membandingkan tiga arsitektur *Deep Learning* untuk menemukan keseimbangan terbaik antara akurasi dan efisiensi komputasi:

1. **Custom CNN**: Arsitektur yang dirancang mandiri (*from scratch*) dengan optimasi *Batch Normalization* dan *Dropout*.
2. **MobileNetV2**: Arsitektur ringan yang dioptimalkan untuk perangkat *mobile* melalui *Inverted Residual Blocks*.
3. **InceptionV3**: Arsitektur kompleks yang menggunakan *Inception Modules* untuk menangkap fitur multi-skala.

---

## 🏗️ Model Architectures

### 1. Custom CNN (Robust Model)
Arsitektur ini terdiri dari 4 blok utama ekstraksi fitur dengan *Learning Rate* 0.0002 untuk menjaga stabilitas di awal pelatihan.


### 2. MobileNetV2 (Lightweight Model)
Menggunakan teknik *Transfer Learning* dan *Fine-Tuning* untuk efisiensi tinggi pada perangkat dengan spesifikasi rendah.


### 3. InceptionV3 (Complex Model)
Memanfaatkan kedalaman jaringan untuk menangkap detail penyakit yang sangat halus pada daun tanaman.


---

## 📂 Repository Structure
```text
.
├── Custom_CNN.ipynb         # Notebook untuk pelatihan Custom CNN
├── InceptionV3.ipynb        # Notebook untuk pelatihan InceptionV3
├── MobileNetV2.ipynb       # Notebook untuk pelatihan MobileNetV2
├── output/                 # Folder hasil analisis
│   ├── models/             # File model (.h5 & .keras via Git LFS)
│   ├── logs/               # Log pelatihan
│   └── plots/              # Grafik Akurasi & Loss
└── .gitattributes          # Konfigurasi Git LFS
