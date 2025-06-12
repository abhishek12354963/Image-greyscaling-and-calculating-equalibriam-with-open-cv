import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("OpenCV Image Processing with Streamlit")

# Upload an image
uploaded_file = st.file_uploader("Upload a car image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image as a NumPy array using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show original image
    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Resize to 500x500
    resized_image = cv2.resize(image, (500, 500))
    st.subheader("Resized Image (500x500)")
    st.image(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Convert to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.subheader("Grayscale Image")
    st.image(gray_image, use_column_width=True, clamp=True)

    # Histogram Equalization
    equalized_image = cv2.equalizeHist(gray_image)
    st.subheader("Histogram Equalized Image")
    st.image(equalized_image, use_column_width=True, clamp=True)
