import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("water_pollution_clean.keras")
    return model

model = load_model()

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.set_page_config(page_title="Water Pollution Detector", layout="centered")

st.title("🌍 Water Pollution Detection")
st.write("AI-powered clean vs polluted water classification (UN SDG 6)")

uploaded = st.file_uploader("Upload a water image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess(img)
        pred = model.predict(x)[0][0]
        label = "🌊 Polluted Water" if pred > 0.5 else "💧 Clean Water"
        confidence = pred if pred > 0.5 else (1 - pred)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

st.markdown("---")
st.caption("Built with TensorFlow • Streamlit • Transfer Learning")
