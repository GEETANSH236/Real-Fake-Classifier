import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
img_base64 = get_base64(r"C:\Users\gtiwa\Desktop\Ai\FakeAi\cnn.png")

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Fake AI Image Detector", layout="wide", page_icon="🧠")

# Add logo path here (Use any PNG logo)
logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/ad/Logo_TV_2015.png"

# =================================================
# CSS (Improved + Clickable Navbar)
# =================================================
st.markdown(
    """
<style>

body {
    background: #0f2027;
    font-family: 'Segoe UI', sans-serif;
}

/* NAVBAR */
.navbar {
    background: rgba(255,255,255,0.05);
    padding: 12px 20px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}

.nav-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-left img {
    width: 45px;
    height: 45px;
    border-radius: 8px;
}

.nav-items {
    display: flex;
    gap: 25px;
}

.nav-item {
    color: #ffdd57;
    font-size: 1.1rem;
    cursor: pointer;
    font-weight: bold;
}

.nav-item:hover {
    color: white;
}

/* HERO SECTION */
.hero {
    text-align: center;
    color: white;
    padding: 40px 10px;
}

.hero h1 {
    font-size: 3.2rem;
    color: #ffdd57;
    font-weight: 800;
}

.hero p {
    font-size: 1.2rem;
    color: #d0d0d0;
}

/* SECTION HEADERS */
.section-title {
    text-align: center;
    font-size: 2.2rem;
    color: #ffdd57;
    margin-top: 40px;
    margin-bottom: 10px;
    font-weight: 700;
}

/* ABOUT CARDS */
.card {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px;
    color: white;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

/* RESULT BOX */
.result-box {
    padding: 15px;
    text-align: center;
    border-radius: 12px;
    margin-top: 10px;
    backdrop-filter: blur(10px);
    font-weight: bold;
    font-size: 1.1rem;
}

.synthetic {background: rgba(230,57,70,0.35); border:1px solid #e63946;}
.human {background: rgba(42,157,143,0.35); border:1px solid #2a9d8f;}

/* FOOTER */
.footer {
    margin-top: 40px;
    background: rgba(255,255,255,0.08);
    color: white;
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    backdrop-filter: blur(12px);
}

</style>
""",
    unsafe_allow_html=True,
)


# HERO SECTION # =================================================
st.markdown("<a id='home'></a>", unsafe_allow_html=True)
st.markdown(
    """ <div class="hero"> <h1>Fake AI Image Detection System</h1> <p>Identify whether an image is Real or AI Generated using a Custom CNN Model.</p> </div> """,
    unsafe_allow_html=True,
)


# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\gtiwa\Desktop\Ai\FakeAi\bestatend.h5")


model = load_model()


# =================================================
# DETECTION SECTION
# =================================================
st.markdown("<a id='detect'></a>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-title'>🔍 Upload Images for Detection</div>",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload one or more images", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
)

if uploaded_files:

    cols_per_row = 3
    threshold = 0.6

    for i in range(0, len(uploaded_files), cols_per_row):
        row = st.columns(cols_per_row)

        for j, col in enumerate(row):
            if i + j < len(uploaded_files):

                file = uploaded_files[i + j]
                img = Image.open(file).convert("RGB")

                resized = img.resize((224, 224))
                arr = np.expand_dims(np.array(resized) / 255.0, axis=0)

                prob = float(model.predict(arr, verbose=0)[0][0])

                if prob > threshold:
                    cls = "Real Image"
                    css = "human"
                    conf = prob
                    emoji = "✅"
                else:
                    cls = "AI-Generated"
                    css = "synthetic"
                    conf = 1 - prob
                    emoji = "⚠️"

                with col:
                    st.image(img, caption=file.name, use_container_width=True)
                    st.markdown(
                        f"<div class='result-box {css}'>{emoji} {cls}<br>Confidence: {conf:.2%}</div>",
                        unsafe_allow_html=True,
                    )


# =================================================
# ABOUT SECTION
st.markdown("<a id='about'></a>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-title'>📘 Overview</div>", unsafe_allow_html=True
)

st.markdown(
    """
<div class="card">
This project focuses on detecting AI-generated fake images using a fine-tuned Convolutional Neural Network.
With the rise in deepfake technology and GAN-generated content, distinguishing between real and synthetic
images has become a major challenge.  
<br><br>
The model learns deep representations, textures, noise patterns, and pixel-level features that commonly appear 
in GAN-based fakes but not in natural images.
</div>
""",
    unsafe_allow_html=True,
)

# Add image safely
st.markdown(
    f"""
    <div style='display: flex; justify-content: center;'>
        <img src='data:image/png;base64,{img_base64}'
             style='max-width: 75%; height: auto; border-radius: 8px; margin-top:50px;' />
    </div>
    """,
    unsafe_allow_html=True
)




# =================================================
# MODEL ARCHITECTURE SECTION
# =================================================
st.markdown("<a id='model'></a>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-title'>🧠 Model Architecture</div>", unsafe_allow_html=True
)

st.markdown(
    """
<div class="card" style="text-align:center;">
A custom Convolutional Neural Network was designed with:
<br><br>
✔ 4 Convolutional layers  
✔ Max Pooling  
✔ Dense layers  
✔ Sigmoid Output  
✔ Trained on Real + AI (GAN) Images  
<br>
<b>Input Size:</b> 224 × 224  
</div>
""",
    unsafe_allow_html=True,
)




# =================================================
# CONTACT SECTION
# =================================================
st.markdown("<a id='contact'></a>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📞 Contact</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="card" style="text-align:center;">
📧 Email: support@example.com<br>
🌐 Website: www.fakeimagedetector.ai<br>
📸 Instagram: @ai_detection_lab
</div>
""",
    unsafe_allow_html=True,
)


# =================================================
# FOOTER
# =================================================
st.markdown(
    """
<div class="footer">
    © 2025 Fake AI Image Detection Project 
</div>
""",
    unsafe_allow_html=True,
)
