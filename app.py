import streamlit as st
import tensorflow as tf
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom, RandomTranslation
# --- ✅ Fix: Set Page Config at the Top ---
st.set_page_config(page_title="Fashion Classifier", layout="centered")

# --- Load the Model & Class Names ---
@st.cache_resource

def load_model():
    model = tf.keras.models.load_model("class_weights_corrected_final_xception_model.keras")  # Load model
    
    # Load class names from JSON file
    try:
        with open("class_weights_corrected_final_class_names.json", "r") as f:
            class_names = json.load(f)
    except FileNotFoundError:
        class_names = ["Unknown"]  # Fallback if missing
    return model, class_names

model, class_names = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    img = np.array(image)  # Convert PIL Image to NumPy array

    # --- Ensure image is RGB (remove Alpha channel if present) ---
    if img.shape[-1] == 4:  # Check if image has 4 channels (RGBA)
        img = img[:, :, :3]  # Keep only the first 3 channels (RGB)

    img = cv2.resize(img, (299, 299))  # Resize for Xception model
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# --- Predict Function ---
def predict_image(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)  # Get highest probability class
    return class_names[predicted_class]  # Return class name

# --- Streamlit UI ---
st.title("👕 Fashion Product Classifier")
st.write("Upload an image to classify it.")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Run Prediction ---
    with st.spinner("Classifying..."):
        predicted_category = predict_image(image)
    
    st.success(f"**Predicted Category:** {predicted_category}")
