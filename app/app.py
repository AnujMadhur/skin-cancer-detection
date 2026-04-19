# app/app.py
import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR       = Path(__file__).parent.parent / "models"
MODEL_PATH      = MODEL_DIR / "best_model.h5"
INFO_PATH       = MODEL_DIR / "class_info.json"
GDRIVE_MODEL_ID = "1-sUCNhyd9biux1T7HJ5bGDSpYs9kqTK0"

# ── MUST BE FIRST st command — no exceptions ──────────────────
st.set_page_config(
    page_title = "Skin Cancer Detector",
    page_icon  = "🔬",
    layout     = "centered"
)

# ── Download model if not present ─────────────────────────────
def download_model_if_needed():
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading model... please wait ~1 minute on first load."):
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}&export=download&confirm=t"
            gdown.download(url, str(MODEL_PATH), quiet=False, fuzzy=True)
        st.success("Model downloaded! Loading now...")

download_model_if_needed()

# ── Load TensorFlow after download ───────────────────────────
import tensorflow as tf

# ── Load model and class info ─────────────────────────────────
@st.cache_resource
def load_model_and_info():
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(INFO_PATH) as f:
        info = json.load(f)
    return model, info

model, class_info = load_model_and_info()

CLASS_NAMES = class_info['class_names']
LABEL_MAP   = class_info['label_map']
DANGEROUS   = class_info['dangerous_classes']
IMG_SIZE    = class_info['img_size']

# ── Preprocess ────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ── UI ────────────────────────────────────────────────────────
st.title("🔬 Skin Cancer Detection")
st.markdown("Upload a **dermatoscopic image** of a skin lesion to get an AI-based classification.")

st.warning(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "It is **NOT** a substitute for professional medical diagnosis. "
    "Always consult a qualified dermatologist."
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** EfficientNetB3  
    **Dataset:** HAM10000 (10,015 images)  
    **Classes:** 7 types of skin lesions  
    **Framework:** TensorFlow / Keras
    """)
    st.header("Classes")
    for code, name in LABEL_MAP.items():
        icon = "🔴" if code in DANGEROUS else "🟢"
        st.markdown(f"{icon} **{code.upper()}** — {name}")

# ── Upload ────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a skin lesion image (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Analysing image..."):
            preds     = model.predict(preprocess(img))[0]
            top_idx   = int(np.argmax(preds))
            top_class = CLASS_NAMES[top_idx]
            top_conf  = float(preds[top_idx]) * 100

        if top_class in DANGEROUS:
            st.error(f"⚠️ **{top_class.upper()}** — {LABEL_MAP[top_class]}")
            st.markdown("**High-risk lesion. Please consult a dermatologist.**")
        else:
            st.success(f"✅ **{top_class.upper()}** — {LABEL_MAP[top_class]}")

        st.metric("Confidence", f"{top_conf:.1f}%")

    st.subheader("All Class Probabilities")
    for idx in np.argsort(preds)[::-1]:
        cls  = CLASS_NAMES[idx]
        prob = float(preds[idx]) * 100
        icon = "🔴" if cls in DANGEROUS else "🟢"
        st.progress(
            prob / 100,
            text=f"{icon} {cls.upper()} — {LABEL_MAP[cls]}  ({prob:.1f}%)"
        )

    interpretations = {
        'akiec': "Actinic keratosis is a pre-cancerous lesion caused by UV exposure. Medical evaluation is recommended.",
        'bcc'  : "Basal cell carcinoma is the most common skin cancer. Highly treatable when caught early.",
        'bkl'  : "Benign keratosis is a harmless growth. No immediate treatment needed but monitor for changes.",
        'df'   : "Dermatofibroma is a benign skin growth, usually harmless.",
        'mel'  : "Melanoma is the most serious form of skin cancer. URGENT: Please see a dermatologist immediately.",
        'nv'   : "Melanocytic nevi (moles) are common and usually harmless. Monitor for changes in size or colour.",
        'vasc' : "Vascular lesion is typically benign. A doctor can confirm and advise if needed."
    }
    st.subheader("What does this mean?")
    st.info(interpretations[top_class])

else:
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Click **Browse files** above  
    2. Upload a clear image of the skin lesion  
    3. The AI will classify it into one of 7 categories  
    4. Review the confidence scores  
    """)
    st.markdown("### The 7 Skin Lesion Types")
    cols = st.columns(2)
    for i, (code, name) in enumerate(LABEL_MAP.items()):
        with cols[i % 2]:
            icon = "🔴" if code in DANGEROUS else "🟢"
            st.markdown(f"{icon} **{code.upper()}** — {name}")