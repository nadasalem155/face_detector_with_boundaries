import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="Live Face Detection", page_icon="📷")
st.title("📷 Live Face Detection with Start/Stop + Upload Image")

# Load face detection model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# --- Video Transformer ---
class FaceDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.5:
                box = detections[0,0,i,3:7] * [w,h,w,h]
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
        return img

# --- Start / Stop Camera ---
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Camera"):
        st.session_state.camera_running = True
with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.camera_running = False

if st.session_state.camera_running:
    webrtc_ctx = webrtc_streamer(
        key="live_face",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceDetector,
        media_stream_constraints={"video": True, "audio": False},  # Disable audio
    )

    if webrtc_ctx.video_transformer and st.button("💾 Save Current Frame"):
        frame = webrtc_ctx.video_transformer.frame
        if frame is not None:
            cv2.imwrite("saved_live_frame.jpg", frame)
            st.success("✅ Live frame saved as saved_live_frame.jpg")

# --- Upload Image Section ---
st.markdown("### Upload an Image")
uploaded_file = st.file_uploader("🖼 Upload an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0, (300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*[w,h,w,h]
            (startX,startY,endX,endY) = box.astype("int")
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Faces")
    if st.button("💾 Save Uploaded Image"):
        cv2.imwrite("saved_uploaded_image.jpg", image)
        st.success("✅ Uploaded image saved as saved_uploaded_image.jpg")