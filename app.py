import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ===============================
# Page Setup
# ===============================
st.set_page_config(
    page_title="TrueVision - Image Authenticity Checker",
    page_icon="🔥",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            color: #ffdd57;
            margin-bottom: 10px;
        }
        .sub {
            text-align: center;
            font-size: 1.1rem;
            color: #cccccc;
            margin-bottom: 30px;
        }
        .result-box {
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: bold;
            margin-top: 8px;
        }
        .synthetic {background-color: #e63946; color: white;}
        .human {background-color: #2a9d8f; color: white;}
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.9rem;
            color: #aaaaaa;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Title + Subtitle
# ===============================
st.markdown("<div class='title'>FAKE DETECTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Detect if an image is <b>AI-generated</b> or <b>Human-made</b>.</div>", unsafe_allow_html=True)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_cnn_model():
    model_path = r"C:\Users\gtiwa\Desktop\Ai\FakeAi\fake_real_cnn1.h5"
    model = load_model(model_path)
    return model

model = load_cnn_model()

# ===============================
# File Upload (Multiple)
# ===============================
uploaded_files = st.file_uploader("📸 Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"📂 {len(uploaded_files)} image(s) uploaded")

    # Layout: 3 images per row
    cols_per_row = 3
    threshold = 0.6

    for i in range(0, len(uploaded_files), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(uploaded_files):
                uploaded_file = uploaded_files[i + j]
                img = Image.open(uploaded_file).convert("RGB")

                # Preprocess for model
                img_resized = img.resize((160, 160))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                pred_prob = float(model.predict(img_array)[0][0])
                if pred_prob > threshold:
                    pred_class = "Real Image"
                    confidence = pred_prob
                    css_class = "human"
                    emoji = "✅"
                else:
                    pred_class = "AI-Generated Image"
                    confidence = 1 - pred_prob
                    css_class = "synthetic"
                    emoji = "⚠️"

                # Display result
                with col:
                    st.image(img, caption=f"{uploaded_file.name}", use_container_width=True)
                    st.markdown(
                        f"<div class='result-box {css_class}'>{emoji} {pred_class}<br>Confidence: {confidence:.2%}</div>",
                        unsafe_allow_html=True
                    )
