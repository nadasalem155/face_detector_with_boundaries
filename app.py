import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_faces(frame):
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

st.set_page_config(page_title="Face Detection App", page_icon="📷")
st.title("📷 Real-Time & Image Face Detection")

# Upload image
uploaded_file = st.file_uploader("🖼 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        pil_image = Image.open(uploaded_file)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame_with_faces = detect_faces(image)
        if frame_with_faces is not None:
            st.image(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB),
                     caption="Detected Faces", use_container_width=True)
        else:
            st.error("⚠ No faces detected in the image!")
    except:
        st.error("⚠ Unable to read the uploaded image!")

# Camera input
st.markdown("---")
st.subheader("📷 Capture an image with camera")
photo = st.camera_input("Click the camera to take a photo")

if photo:
    try:
        pil_image = Image.open(photo)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame_with_faces = detect_faces(image)
        if frame_with_faces is not None:
            st.image(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB),
                     caption="Detected Faces", use_container_width=True)
        else:
            st.error("⚠ No faces detected in the photo!")
    except:
        st.error("⚠ Unable to read the captured photo!")