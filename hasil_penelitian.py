import streamlit as st
import pandas as pd

def render():
    st.title("📊 Hasil Penelitian")

    st.subheader("Tabel Hasil Pelatihan Custom CNN")
    df_cnn = pd.DataFrame({
        "Batch Size": [8, 16, 32],
        "Epoch Berhenti": [47, 46, 44],
        "Val. Accuracy": ["87,56%", "92,59%", "92,77%"],
        "Val. Loss": [0.4884, 0.2538, 0.2577],
        "F1-Score (Macro)": [0.88, 0.92, 0.92]
    })
    st.table(df_cnn)

    st.subheader("Tabel Hasil Pelatihan MobileNetV2")
    df_mobilenet = pd.DataFrame({
        "Batch Size": [8, 16, 32],
        "Val. Accuracy": ["90,14%", "93,31%", "93,31%"],
        "Val. Loss": [0.3541, 0.1935, 0.1935],
        "F1-Score (Macro)": [0.93, 0.93, 0.94]
    })
    st.table(df_mobilenet)

    st.subheader("Tabel Hasil Pelatihan InceptionV3")
    df_inception = pd.DataFrame({
        "Batch Size": [8, 16, 32],
        "Val. Accuracy": ["92,72%", "92,72%", "92,72%"],
        "Val. Loss": [0.2131, 0.2131, 0.2131],
        "F1-Score (Macro)": [0.93, 0.93, 0.92]
    })
    st.table(df_inception)

    st.subheader("Tabel Rekap Perbandingan")
    df_rekap = pd.DataFrame({
        "Metrik": ["Val. Accuracy", "Val. Loss", "F1-Score (Macro)"],
        "Custom CNN (BS 32)": ["92,77%", 0.2577, 0.92],
        "MobileNetV2 (BS 16)": ["93,31%", 0.1935, 0.93],
        "InceptionV3 (BS 16)": ["92,72%", 0.2131, 0.93]
    })
    st.table(df_rekap)
