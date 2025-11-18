import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="water_pollution_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img):
    img = img.resize((224,224))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "🌊 Polluted Water" if pred > 0.5 else "💧 Clean Water"
    confidence = pred if pred > 0.5 else (1 - pred)

    return label, float(confidence)

st.title("🌍 Water Pollution Detector")
uploaded = st.file_uploader("Upload water image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, conf = predict(img)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf*100:.2f}%")
