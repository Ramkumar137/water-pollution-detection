import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Water Pollution Detector", layout="centered")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("water_pollution_clean.keras")
    return model

model = load_model()

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------------------
# UI
# ---------------------------
st.title("🌍 Water Pollution Detection System")
st.write(
    """
    This AI-powered image classifier detects whether a water body is **Clean** or **Polluted**.
    Built using **MobileNetV2 + Transfer Learning** aligned with **UN SDG 6 — Clean Water & Sanitation**.
    """
)

uploaded_file = st.file_uploader("Upload an image of a water body", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess(img)
        prediction = model.predict(x)[0][0]

        label = "🌊 Polluted Water" if prediction > 0.5 else "💧 Clean Water"
        confidence = prediction if prediction > 0.5 else (1 - prediction)

        st.subheader(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")

st.markdown("---")
st.caption("Developed by Ram • Powered by TensorFlow + Streamlit")
