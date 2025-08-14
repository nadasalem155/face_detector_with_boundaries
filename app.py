import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Face Detection", page_icon="📷")
st.title("📷 Live Face Detection + Upload Image")

# Load face detection model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Function to detect faces
def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
    return frame

# --- Upload Image Section ---
st.markdown("### Upload an Image")
uploaded_file = st.file_uploader("🖼 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = detect_faces(image)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Faces")
    if st.button("💾 Save Uploaded Image"):
        cv2.imwrite("saved_uploaded_image.jpg", image)
        st.success("✅ Uploaded image saved as saved_uploaded_image.jpg")

# --- Live Camera Section ---
st.markdown("### Or use your camera live")

class FaceDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = detect_faces(img)
        self.last_frame = img  # حفظ آخر frame
        return img

webrtc_ctx = webrtc_streamer(
    key="live_camera",
    video_transformer_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},  # الصوت اتعطل
)

# Save frame from live camera
if webrtc_ctx.video_transformer:
    if st.button("💾 Save Current Frame"):
        frame = webrtc_ctx.video_transformer.last_frame
        if frame is not None:
            cv2.imwrite("saved_face_image.jpg", frame)
            st.success("✅ Current frame saved as saved_face_image.jpg")