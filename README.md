# 🔬 Skin Cancer Detection using Deep Learning

An end-to-end deep learning project to classify skin lesions from dermatoscopic images using the HAM10000 dataset and EfficientNetB3 transfer learning.

## 🎯 Problem Statement
Early detection of skin cancer is critical for effective treatment. This project builds an automated classifier that identifies 7 types of skin lesions from images.

## 📊 Dataset
- **Name:** HAM10000 (Human Against Machine with 10,000 images)
- **Source:** Harvard Dataverse / Kaggle
- **Size:** 10,015 images across 7 classes
- **Classes:** akiec, bcc, bkl, df, mel, nv, vasc

## 🏗️ Architecture
- **Base Model:** EfficientNetB3 (pre-trained on ImageNet)
- **Technique:** Transfer Learning + Fine-tuning
- **Framework:** TensorFlow / Keras

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
streamlit run app/app.py
```

## 📁 Project Structure