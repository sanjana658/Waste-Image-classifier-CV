import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("waste_classifier.h5")

# 🔥 Load correct class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("♻️ Waste Image Classification")
st.write("Upload a waste image and the model will classify it.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.success(f"Prediction: {predicted_class.replace('_', ' ').title()}")
    st.info(f"Confidence: {confidence:.2f}%")
