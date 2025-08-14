import cv2
import numpy as np
import streamlit as st

# Load face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_faces(frame):
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

st.set_page_config(page_title="Face Detection", page_icon="📷")
st.title("📷 Face Detection (Upload + Camera)")

# ---------------- Upload Image ----------------
st.markdown("### 🖼 Upload an Image")
uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_with_faces = detect_faces(image.copy())
    st.image(cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_container_width=True)
    if st.button("💾 Save Uploaded Image"):
        cv2.imwrite("uploaded_image_with_faces.jpg", image_with_faces)
        st.success("✅ Uploaded image saved!")

# ---------------- Camera Input ----------------
st.markdown("### 🎥 Live Camera")
camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame_with_faces = detect_faces(frame.copy())
    st.image(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_container_width=True)
    if st.button("💾 Save Camera Photo"):
        cv2.imwrite("camera_image_with_faces.jpg", frame_with_faces)
        st.success("✅ Camera image saved!")