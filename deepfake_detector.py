import os
import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.title("🔍 DeepFake Detector App")

# Load pre-trained DeepFake detection model
MODEL_PATH = r"E:\Study Material\Internship\deepfake-detection-model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("❌ DeepFake model not found! Please add 'deepfake_model.h5' in the same directory.")
else:
    model = load_model(MODEL_PATH)
    st.success("✅ DeepFake Detection Model Loaded!")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize for model input (assuming 224x224)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        try:
            # DeepFake classification using AI model
            prediction = model.predict(img_array)
            fake_prob = prediction[0][0]

            # Display results
            if fake_prob > 0.5:
                st.error("🔴 This is a **FAKE** (DeepFake) image!")
            else:
                st.success("🟢 This is an **ORIGINAL** image!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
