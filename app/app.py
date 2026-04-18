# app/app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import gdown
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Skin Cancer Detector",
    page_icon  = "🔬",
    layout     = "centered"
)

# ── Load model and class info ─────────────────────────────────
@st.cache_resource
def load_model_and_info():
    base = Path(__file__).parent.parent
    model_dir = base / "models"
    model_path = model_dir / "best_model.h5"
    info_path = model_dir / "class_info.json"

    # create folder
    model_dir.mkdir(exist_ok=True)

    # Download model if not present
    if not model_path.exists():
        file_id = "1-sUCNhyd9biux1T7HJ5bGDSpYs9kqTK0"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(model_path), quiet=False)

    model = tf.keras.models.load_model(str(model_path), compile=False)

    with open(info_path) as f:
        info = json.load(f)

    return model, info

model, class_info = load_model_and_info()
CLASS_NAMES = class_info['class_names']
LABEL_MAP   = class_info['label_map']
DANGEROUS   = class_info['dangerous_classes']
IMG_SIZE    = class_info['img_size']

# ── Preprocess image ──────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)

# ── UI ────────────────────────────────────────────────────────
st.title("🔬 Skin Cancer Detection")
st.markdown("Upload a **dermatoscopic image** of a skin lesion to get an AI-based classification.")

st.warning(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "It is **NOT** a substitute for professional medical diagnosis. "
    "Always consult a qualified dermatologist."
)

# ── Sidebar info ──────────────────────────────────────────────
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
        color = "🔴" if code in DANGEROUS else "🟢"
        st.markdown(f"{color} **{code.upper()}** — {name}")

# ── Upload section ─────────────────────────────────────────────
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
            input_arr = preprocess(img)
            preds = model.predict(input_arr)[0]
        
        top_idx   = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        top_conf  = float(preds[top_idx]) * 100
        
        is_dangerous = top_class in DANGEROUS
        
        if is_dangerous:
            st.error(f"⚠️ **{top_class.upper()}** — {LABEL_MAP[top_class]}")
            st.markdown("**High-risk lesion detected. Please consult a dermatologist.**")
        else:
            st.success(f"✅ **{top_class.upper()}** — {LABEL_MAP[top_class]}")
        
        st.metric("Confidence", f"{top_conf:.1f}%")
    
    # ── All class probabilities ─────────────────────────────────
    st.subheader("All Class Probabilities")
    
    # Sort by probability (highest first)
    sorted_indices = np.argsort(preds)[::-1]
    
    for idx in sorted_indices:
        cls  = CLASS_NAMES[idx]
        prob = float(preds[idx]) * 100
        color = "🔴" if cls in DANGEROUS else "🟢"
        label = f"{color} {cls.upper()} — {LABEL_MAP[cls]}"
        st.progress(prob / 100, text=f"{label}  ({prob:.1f}%)")
    
    # ── Interpretation ──────────────────────────────────────────
    st.subheader("What does this mean?")
    interpretations = {
        'akiec': "Actinic keratosis is a pre-cancerous lesion caused by UV exposure. Medical evaluation is recommended.",
        'bcc'  : "Basal cell carcinoma is the most common skin cancer. Highly treatable when caught early.",
        'bkl'  : "Benign keratosis is a harmless growth. No immediate treatment needed but monitor for changes.",
        'df'   : "Dermatofibroma is a benign skin growth, usually harmless.",
        'mel'  : "Melanoma is the most serious form of skin cancer. URGENT: Please see a dermatologist immediately.",
        'nv'   : "Melanocytic nevi (moles) are common and usually harmless. Monitor for changes in size or colour.",
        'vasc' : "Vascular lesion is typically benign. A doctor can confirm and advise on treatment if needed."
    }
    st.info(interpretations[top_class])

else:
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Click **Browse files** above
    2. Upload a clear image of the skin lesion
    3. The AI will classify it into one of 7 categories
    4. Review the confidence scores for all classes
    """)
    
    # Show sample class info
    st.markdown("### The 7 Skin Lesion Types")
    cols = st.columns(2)
    for i, (code, name) in enumerate(LABEL_MAP.items()):
        with cols[i % 2]:
            color = "🔴" if code in DANGEROUS else "🟢"
            st.markdown(f"{color} **{code.upper()}** — {name}")