import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===============================================
# Page Setup
# ===============================================
st.set_page_config(
    page_title="AI IMAGE CLASSIFIER",
    page_icon="🔥",
    layout="wide"
)

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
    </style>
""", unsafe_allow_html=True)

# ===============================================
# Title
# ===============================================
st.markdown("<div class='title'>FAKE DETECTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Detect if an image is <b>AI-generated</b> or <b>Human-made</b>.</div>", unsafe_allow_html=True)

# ===============================================
# Load H5 Model (Keras)
# ===============================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\gtiwa\Desktop\Ai\FakeAi\bestatend.h5")
    return model

model = load_model()

# ===============================================
# File Upload
# ===============================================
uploaded_files = st.file_uploader(
    "📸 Upload one or more images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"📂 {len(uploaded_files)} image(s) uploaded")

    cols_per_row = 3
    threshold = 0.6

    for i in range(0, len(uploaded_files), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(uploaded_files):
                uploaded_file = uploaded_files[i + j]
                img = Image.open(uploaded_file).convert("RGB")

                # PREPROCESS
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # H5 MODEL PREDICTION
                pred_prob = float(model.predict(img_array, verbose=0)[0][0])

                # CLASSIFICATION
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

                # DISPLAY RESULT
                with col:
                    st.image(img, caption=uploaded_file.name, use_container_width=True)
                    st.markdown(
                        f"""
                        <div class='result-box {css_class}'>
                            {emoji} {pred_class}<br>
                            Confidence: {confidence:.2%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
