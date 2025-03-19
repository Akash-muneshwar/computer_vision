import cv2
import streamlit as st
import numpy as np
from PIL import Image
import time  # For adding delay

# Streamlit UI
st.title("🔍 Number Plate Recognition")
st.write("Upload an image to detect the number plate.")

# Load Haar cascade
nPlateCascade = cv2.CascadeClassifier("/home/user/Downloads/haarcascade_russian_plate_number.xml")
minArea = 200
color = (255, 0, 255)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.resize(img, (640, 480))

    # Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Show spinner only while detecting number plates
    with st.spinner("🔄 Processing image..."):
        time.sleep(2)  # Simulate processing time
        numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)

    # Process detected plates
    imgRoi = None
    if len(numberPlates) == 0:
        st.image(img)
        st.error("🚫 No Number Plate Detected")
    else:
        for (x, y, w, h) in numberPlates:
            area = w * h
            if area > minArea:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
                imgRoi = img[y:y + h, x:x + w]

        # Display the result
        st.image(img, caption="Detected Number Plate", channels="RGB")

        # Show extracted number plate with a delay
        if imgRoi is not None:
            with st.spinner("🛠️ Extracting Number Plate..."):
                time.sleep(1.5)  # Simulate extraction time
                st.image(imgRoi, caption="Extracted Number Plate", channels="RGB")

            # Save option
            if st.button("Save Number Plate"):
                cv2.imwrite("NoPlate.jpg", imgRoi)
                st.success("✅ Number plate saved successfully!")
