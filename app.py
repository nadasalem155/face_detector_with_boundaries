import cv2
import numpy as np
import streamlit as st
import time

# Load face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Function to detect faces
def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

# Streamlit layout
st.set_page_config(page_title="Face Detection App", page_icon="📷")
st.title("📷 Real-Time & Image Face Detection")

st.markdown("""
<style>
.stButton>button {
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stFileUploader {
    border: 2px dashed #4CAF50;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# Upload image section
uploaded_file = st.file_uploader("🖼 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = detect_faces(image)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Faces")

# Camera section
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Start Camera"):
        st.session_state.camera_running = True
with col2:
    if st.button("🛑 Stop Camera"):
        st.session_state.camera_running = False

if st.session_state.camera_running:
    with st.spinner("Loading camera..."):
        time.sleep(0.5)  # small delay to show spinner
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
            frame = detect_faces(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if not st.session_state.camera_running:
                break
        cap.release()