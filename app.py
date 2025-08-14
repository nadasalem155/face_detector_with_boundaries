import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Face Detector DNN", page_icon="👤")

# ---- Load the DNN model directly from local files ----
proto_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# ---- Helper functions ----
def pil_to_bgr(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img.convert("RGB"))
    return np.array(pil_img)[:, :, ::-1].copy()  # RGB -> BGR

def bgr_to_pil(bgr_img):
    return Image.fromarray(bgr_img[:, :, ::-1])  # BGR -> RGB

def detect_faces(image_bgr, conf_threshold=0.6):
    (h, w) = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startX, startY, endX, endY))
    return boxes

def draw_boxes(image_bgr, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_bgr

# ---- Streamlit UI ----
st.title("👤 Face Detection using OpenCV DNN")
st.write("Upload a photo or take a snapshot to detect faces.")

confidence = st.slider("Confidence Threshold", 0.1, 0.99, 0.6, 0.01)

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Take Photo"])

with tab1:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        bgr = pil_to_bgr(img)
        faces = detect_faces(bgr, conf_threshold=confidence)
        out_img = draw_boxes(bgr.copy(), faces)
        st.image(bgr_to_pil(out_img), caption=f"Detected {len(faces)} face(s)")
        buf = io.BytesIO()
        bgr_to_pil(out_img).save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), "faces.png", "image/png")

with tab2:
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        img = Image.open(cam_img)
        bgr = pil_to_bgr(img)
        faces = detect_faces(bgr, conf_threshold=confidence)
        out_img = draw_boxes(bgr.copy(), faces)
        st.image(bgr_to_pil(out_img), caption=f"Detected {len(faces)} face(s)")
        buf = io.BytesIO()
        bgr_to_pil(out_img).save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), "faces.png", "image/png")