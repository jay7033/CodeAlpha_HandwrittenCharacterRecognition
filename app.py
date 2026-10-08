import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = "models/handwritten_digit_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and let the CNN model predict it.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array, verbose=0)

    predicted_digit = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")

    probabilities = prediction[0] * 100

    for digit, probability in enumerate(probabilities):
        st.write(f"Digit {digit}: {probability:.2f}%")