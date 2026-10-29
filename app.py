import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Classification")
st.write("Upload a waste image to classify it using a CNN.")

MODEL_PATH = Path("waste_classifier.keras")
CLASS_PATH = Path("classes.txt")

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error("Model files are missing. Run python train.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = CLASS_PATH.read_text(
    encoding="utf-8"
).splitlines()

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=350)

    if st.button("Classify Waste", type="primary"):
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        probability = float(model.predict(image_array, verbose=0)[0][0])

        # Binary labels: index 0 -> sigmoid output below 0.5
        # index 1 -> sigmoid output at or above 0.5
        predicted_index = 1 if probability >= 0.5 else 0
        confidence = (
            probability if predicted_index == 1
            else 1.0 - probability
        )

        predicted_class = class_names[predicted_index]

        st.subheader("Prediction Result")
        st.success(f"Predicted category: {predicted_class}")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")

        st.caption(
            "Predictions can be incorrect, especially for unfamiliar "
            "images. Confidence is not a guarantee of correctness."
        )