import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Water Pollution Detector", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("water_pollution_model")
    return model

model = load_model()

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

st.title("🌍 Water Pollution Detection")
st.write("Upload an image to classify clean vs polluted water.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess(img)
        pred = model.predict(x)[0][0]

        label = "🌊 Polluted Water" if pred > 0.5 else "💧 Clean Water"
        conf = pred if pred > 0.5 else (1 - pred)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf * 100:.2f}%")
