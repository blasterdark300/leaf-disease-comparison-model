import os
import tempfile
import cv2
from io import BytesIO
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import streamlit as st
import tensorflow as tf

# =========================================================
# KONFIGURASI MODEL & SAMPLE IMAGES
# =========================================================
MODELS_CONFIG = {
    "MobileNetV2": {
        "model_url": "https://github.com/blasterdark300/leaf-disease-comparison-model/raw/main/8/(8)%20mobilenetv2%20811/models/Leaf_Disease_MobileNetV2_FIXED_v3.keras",
        "labels_url": "https://raw.githubusercontent.com/blasterdark300/leaf-disease-comparison-model/main/8/(8)%20mobilenetv2%20811/labels/class_indices.json",
        "img_size": (256, 256),
        "preprocess": "rescale",
        "icon": "📱",
        "accent": "#2D6A4F",
    },
    "InceptionV3": {
        "model_url": "https://github.com/blasterdark300/leaf-disease-comparison-model/raw/main/8/(8)%20inceptionv3%20811/models/Leaf_Disease_InceptionV3_FIXED_v3.h5",
        "labels_url": "https://raw.githubusercontent.com/blasterdark300/leaf-disease-comparison-model/main/8/(8)%20inceptionv3%20811/labels/class_indices.json",
        "img_size": (256, 256),
        "preprocess": "inception",
        "icon": "🔬",
        "accent": "#264653",
    },
    "Custom CNN (Scratch)": {
        "model_url": "https://github.com/blasterdark300/leaf-disease-comparison-model/raw/main/32/80%2010%2010/(32)%20Custom%20CNN%20%20811/models/Leaf_Disease_CustomCNN_Scratch_1Fase.h5",
        "labels_url": "https://raw.githubusercontent.com/blasterdark300/leaf-disease-comparison-model/main/32/80%2010%2010/(32)%20Custom%20CNN%20%20811/labels/class_indices.json",
        "img_size": (256, 256),
        "preprocess": "rescale",
        "icon": "🧠",
        "accent": "#BC6C25",
    },
}

# URL Folder Gambar Contoh
SAMPLE_IMG_FOLDER_API = "https://api.github.com/repos/blasterdark300/leaf-disease-comparison-model/contents/img"
SAMPLE_IMG_RAW_BASE = "https://raw.githubusercontent.com/blasterdark300/leaf-disease-comparison-model/main/img/"

st.set_page_config(
    page_title="Klasifikasi Penyakit Daun",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS — OPTIMALISASI MOBILE & TAMPILAN
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-tap-highlight-color: transparent;
        }

        * { box-sizing: border-box; }

        .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li,
        .streamlit-expanderHeader, .streamlit-expanderHeader *,
        .streamlit-expanderContent, .streamlit-expanderContent *,
        [data-testid="stExpander"], [data-testid="stExpander"] *,
        [data-testid="stExpanderDetails"], [data-testid="stExpanderDetails"] *,
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
        [data-testid="stText"], [data-testid="stCaptionContainer"] {
            color: #1B2E22 !important;
        }

        .stApp {
            background: linear-gradient(180deg, #FAF8F2 0%, #EEF3E9 55%, #E8F0E6 100%);
        }

        .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 3rem;
            margin: 0 auto;
            width: 100%;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        img { max-width: 100%; height: auto; }
        [data-testid="stImage"] img {
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(27, 67, 50, 0.15);
            margin: 0 auto;
            display: block;
        }

        .hero-box {
            background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 45%, #52B788 100%);
            padding: 1.8rem 1.2rem;
            border-radius: 20px;
            color: #F4FAF6;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 28px rgba(27, 67, 50, 0.22);
            text-align: center;
        }
        .hero-box h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
            line-height: 1.2;
            color: #FFFFFF !important;
        }
        .hero-box p {
            font-size: 0.92rem;
            opacity: 0.95;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.45;
            color: #EAF6EE !important;
        }

        .section-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.15rem;
            color: #1B4332;
            margin: 1.2rem 0 0.6rem 0;
        }

        .model-pill {
            display: inline-block;
            background: #FFFFFF;
            border: 1px solid #C9DFC7;
            color: #2D6A4F;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 2px 4px 4px 0;
            box-shadow: 0 2px 6px rgba(27, 67, 50, 0.05);
        }

        .result-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.1rem 0.9rem;
            box-shadow: 0 6px 20px rgba(27, 67, 50, 0.08);
            border-top: 5px solid var(--accent, #2D6A4F);
            text-align: center;
            height: 100%;
            margin-bottom: 0.5rem;
        }
        .result-card .model-name {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 0.9rem;
            color: #3A5A4A;
            margin-bottom: 0.4rem;
        }
        .result-card .pred-label {
            font-size: 1rem;
            font-weight: 700;
            color: #143A2A;
            background: linear-gradient(135deg, #EAF6EE 0%, #DEF0E3 100%);
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 0.5rem;
            word-break: break-word;
        }
        .result-card .confidence-text {
            font-size: 0.82rem;
            color: #5B7A67;
        }

        .conf-bar-bg {
            background: #E7EFEA;
            border-radius: 8px;
            height: 8px;
            width: 100%;
            margin-top: 6px;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 8px;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #1B4332, #40916C);
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 14px;
            padding: 0.8rem 1.2rem;
            min-height: 48px;
            box-shadow: 0 6px 18px rgba(27, 67, 50, 0.25);
            transition: all 0.2s ease;
            width: 100%;
            font-size: 0.98rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            flex-wrap: nowrap;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 10px 10px 0 0;
            padding: 8px 12px;
            font-size: 0.88rem;
            font-weight: 500;
            white-space: nowrap;
            flex-shrink: 0;
        }

        [data-testid="stExpander"] {
            background: white !important;
            border: 1px solid #C9DFC7;
            border-radius: 14px;
            margin-bottom: 10px;
        }
        [data-testid="stExpanderDetails"] {
            background: white !important;
            padding: 0.8rem;
        }

        @media (max-width: 640px) {
            .block-container { padding-left: 0.6rem; padding-right: 0.6rem; padding-top: 1rem; }
            .hero-box { padding: 1.3rem 0.9rem; border-radius: 16px; }
            .hero-box h1 { font-size: 1.3rem; }
            .hero-box p { font-size: 0.82rem; }
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 0.6rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPER LOADER MODEL, LABELS, & SAMPLE IMAGES
# =========================================================
@st.cache_data(ttl=3600)
def fetch_sample_images():
    """Mengambil daftar gambar contoh dari folder /img di repository GitHub."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(SAMPLE_IMG_FOLDER_API, headers=headers, timeout=10)
        res.raise_for_status()
        files = res.json()
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        image_list = []
        for file in files:
            if file.get("type") == "file" and file.get("name", "").lower().endswith(valid_extensions):
                image_list.append({
                    "name": file["name"],
                    "raw_url": file["download_url"]
                })
        return image_list
    except Exception:
        # Fallback jika GitHub API rate-limited
        return []

@st.cache_resource
def load_class_names(labels_url: str):
    response = requests.get(labels_url, timeout=15)
    response.raise_for_status()
    data = response.json()

    sample_value = next(iter(data.values()))
    if isinstance(sample_value, int):
        sorted_items = sorted(data.items(), key=lambda x: x[1])
        names = [name for name, idx in sorted_items]
    else:
        sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
        names = [name for idx, name in sorted_items]

    return names


@st.cache_resource
def load_model_cached(model_url: str):
    ext = ".keras" if model_url.endswith(".keras") else ".h5"
    
    response = requests.get(model_url, stream=True, allow_redirects=True, timeout=120)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                tmp_file.write(chunk)
        tmp_path = tmp_file.name

    file_size = os.path.getsize(tmp_path)
    if file_size < 10 * 1024:
        with open(tmp_path, "r", errors="ignore") as f:
            content_preview = f.read(200)
        os.remove(tmp_path)
        raise ValueError(
            f"File yang di-download bukan file model biner valid ({file_size} bytes). "
            f"Gagal resolve Git LFS atau 404 Not Found. Preview isi: {content_preview}"
        )

    try:
        model = tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return model


def preprocess_image(img: Image.Image, img_size, method: str) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img).astype("float32")

    if method == "inception":
        arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    else:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)
    return arr


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name

    raise ValueError("Tidak ditemukan layer Conv2D pada model untuk Grad-CAM.")


def generate_gradcam(model, img_array, orig_img: Image.Image, pred_index=None, alpha=0.4):
    try:
        last_conv_layer_name = find_last_conv_layer(model)

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        orig_img_np = np.array(orig_img.convert("RGB"))
        heatmap_resized = cv2.resize(
            heatmap, (orig_img_np.shape[1], orig_img_np.shape[0])
        )

        jet = plt.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[(heatmap_resized * 255).astype(np.uint8)]
        jet_heatmap = (jet_heatmap * 255).astype(np.uint8)

        superimposed_img = jet_heatmap * alpha + orig_img_np
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return Image.fromarray(superimposed_img)
    except Exception as e:
        return None


def predict(model, class_names, img: Image.Image, img_size, preprocess_method):
    x = preprocess_image(img, img_size, preprocess_method)
    preds = model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
    confidence = float(probs[idx])

    gradcam_img = generate_gradcam(model, x, img, pred_index=idx)

    return label, confidence, probs, gradcam_img


def confidence_color(conf: float) -> str:
    if conf >= 0.75:
        return "#2D6A4F"
    elif conf >= 0.4:
        return "#BC6C25"
    else:
        return "#9C3D1F"


# =========================================================
# UI HEADER & PRELOAD
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <h1>🌿 Klasifikasi Penyakit Daun</h1>
        <p>Gunakan kamera HP, upload dari galeri iPhone/Android, tautan web, atau pilih sampel gambar untuk mendeteksi penyakit daun.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

loaded_models = {}
load_errors = {}

with st.spinner("Memuat model AI dari repository... (Membutuhkan beberapa detik saat pertama kali dipanggil)"):
    for model_name, config in MODELS_CONFIG.items():
        try:
            model = load_model_cached(config["model_url"])
            class_names = load_class_names(config["labels_url"])
            loaded_models[model_name] = {
                "model": model,
                "class_names": class_names,
                "config": config,
            }
        except Exception as e:
            load_errors[model_name] = str(e)

if load_errors:
    st.warning("⚠️ Beberapa model gagal dimuat:")
    for model_name, err in load_errors.items():
        st.error(f"**{model_name}**: {err}")

if not loaded_models:
    st.error("❌ Tidak ada model yang berhasil dimuat. Periksa tautan URL model.")
    st.stop()

pills_html = "".join(
    f'<span class="model-pill">{cfg["icon"]} {name}</span>'
    for name, cfg in MODELS_CONFIG.items()
    if name in loaded_models
)
st.markdown(
    f'<div style="margin-bottom: 0.6rem; text-align: center;">{pills_html}</div>',
    unsafe_allow_html=True,
)

# =========================================================
# INPUT GAMBAR
# =========================================================
st.markdown(
    '<div class="section-title">📤 Pilih Input Gambar</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Upload File / Galeri", "📷 Kamera HP", "🔗 Link URL", "🖼️ Gambar Contoh (/img)"]
)

image_to_predict = None

with tab1:
    uploaded_file = st.file_uploader(
        "Pilih foto dari galeri HP atau dokumen",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="uploader",
    )
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)

with tab2:
    st.caption("💡 *Di iPhone/Android, pastikan memberikan izin akses kamera pada browser.*")
    camera_file = st.camera_input("Ambil foto daun", key="mobile_camera")
    if camera_file is not None:
        image_to_predict = Image.open(camera_file)

with tab3:
    url = st.text_input("Tempel link URL gambar")
    if url:
        try:
            response = requests.get(url, timeout=10)
            image_to_predict = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Gagal memuat URL: {e}")

with tab4:
    st.caption("Pilih salah satu gambar sampel langsung dari folder `/img` repository:")
    sample_images = fetch_sample_images()
    
    if sample_images:
        # Tampilkan gambar dalam bentuk Grid (3 kolom)
        cols_per_row = 3
        for i in range(0, len(sample_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(sample_images):
                    img_item = sample_images[i + j]
                    with col:
                        st.image(img_item["raw_url"], use_column_width=True)
                        if st.button(f"Gunakan {img_item['name']}", key=f"btn_sample_{i+j}"):
                            try:
                                resp = requests.get(img_item["raw_url"], timeout=10)
                                image_to_predict = Image.open(BytesIO(resp.content))
                                st.session_state["selected_sample_img"] = img_item["raw_url"]
                            except Exception as ex:
                                st.error(f"Gagal memuat gambar sampel: {ex}")
        
        # Jaga state jika gambar dari sampel sudah dipilih sebelumnya
        if "selected_sample_img" in st.session_state and image_to_predict is None:
            try:
                resp = requests.get(st.session_state["selected_sample_img"], timeout=10)
                image_to_predict = Image.open(BytesIO(resp.content))
            except Exception:
                pass
    else:
        st.info("ℹ️ Tidak dapat mengambil daftar gambar dari folder `/img` atau folder kosong.")

# =========================================================
# EXECUTION & RESULTS
# =========================================================
if image_to_predict is not None:
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.image(
            image_to_predict,
            caption=f"Ukuran: {image_to_predict.size[0]}x{image_to_predict.size[1]} px",
            use_column_width=True,
        )

    st.write("")
    run_prediction = st.button("🔍 Analisis Sekarang", type="primary")

    if run_prediction:
        results = {}
        with st.spinner("Menganalisis gambar..."):
            for model_name, item in loaded_models.items():
                cfg = item["config"]
                label, confidence, probs, gradcam_img = predict(
                    item["model"],
                    item["class_names"],
                    image_to_predict,
                    cfg["img_size"],
                    cfg["preprocess"],
                )
                results[model_name] = {
                    "label": label,
                    "confidence": confidence,
                    "probs": probs,
                    "gradcam": gradcam_img,
                    "class_names": item["class_names"],
                }

        st.markdown(
            '<div class="section-title">📊 Hasil Diagnosa AI</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(len(results))
        for col, (model_name, res) in zip(cols, results.items()):
            cfg = MODELS_CONFIG[model_name]
            bar_color = confidence_color(res["confidence"])
            with col:
                st.markdown(
                    f"""
                    <div class="result-card" style="--accent: {cfg['accent']};">
                        <div class="model-name">{cfg['icon']} {model_name}</div>
                        <div class="pred-label">{res['label']}</div>
                        <div class="confidence-text">Akurasi: <b>{res['confidence']*100:.2f}%</b></div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{min(max(res['confidence']*100,0),100):.1f}%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        st.markdown(
            '<div class="section-title">🔥 Area Fokus AI (Grad-CAM)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Bagian berwarna **merah/kuning** merupakan fokus utama AI dalam mengambil keputusan.")

        grad_cols = st.columns(len(results))
        for col, (model_name, res) in zip(grad_cols, results.items()):
            cfg = MODELS_CONFIG[model_name]
            with col:
                if res["gradcam"] is not None:
                    st.image(
                        res["gradcam"],
                        caption=f"Grad-CAM: {model_name}",
                        use_column_width=True,
                    )
                else:
                    st.warning(f"Grad-CAM tidak tersedia untuk {model_name}")

        st.divider()

        st.markdown(
            '<div class="section-title">🔎 Rincian Probabilitas</div>',
            unsafe_allow_html=True,
        )
        for model_name, res in results.items():
            cfg = MODELS_CONFIG[model_name]
            with st.expander(
                f"{cfg['icon']} Probabilitas — {model_name}",
                expanded=False,
            ):
                for i, cname in enumerate(res["class_names"]):
                    p = float(res["probs"][i]) if i < len(res["probs"]) else 0.0
                    bar_color = confidence_color(p)
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#3A5A4A;">
                                <span>{cname}</span><span><b>{p*100:.2f}%</b></span>
                            </div>
                            <div class="conf-bar-bg">
                                <div class="conf-bar-fill" style="width:{min(max(p*100,0),100):.1f}%; background:{bar_color};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
else:
    st.info("💡 Silakan upload file dari galeri, ambil foto dari kamera, masukkan link URL, atau pilih contoh gambar dari tab 🖼️ Gambar Contoh.")
