import streamlit as st
import cv2
import numpy as np
from PIL import Image

# App configuration
st.set_page_config(
    page_title="Human Face Identification",
    page_icon="🙂",
    layout="centered"
)

st.title("🧠 Human Face Identification App")
st.write("Upload an image and adjust parameters to detect human faces.")

# Sidebar controls
st.sidebar.header("Detection Parameters")
scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.05,
    max_value=1.5,
    value=1.1,
    step=0.05
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

# Load Haar Cascade
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = load_cascade()

# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("📷 Uploaded Image Preview")
    st.image(image, use_container_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show result
    st.subheader("✅ Detection Result")
    st.image(img_array, use_container_width=True)

    st.success(f"Total Faces Detected: {len(faces)}")

else:
    st.info("Please upload an image to start face detection.")
