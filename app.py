import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Face Detection · Streamlit", page_icon="👤", layout="centered")

PROTOTXT_NAME = "deploy.prototxt"
CAFFE_NAME = "res10_300x300_ssd_iter_140000.caffemodel"

def find_path(filename: str) -> str:
    candidates = [filename, os.path.join("models", filename)]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

proto_path = find_path(PROTOTXT_NAME)
caffe_path = find_path(CAFFE_NAME)

if not proto_path or not caffe_path:
    st.error(
        "Missing files! Please place *deploy.prototxt* and "
        "*res10_300x300_ssd_iter_140000.caffemodel* next to app.py "
        "or inside a folder named models/."
    )
    st.stop()

# --------- Load network once ---------
@st.cache_resource(show_spinner=True)
def load_net():
    return cv2.dnn.readNetFromCaffe(proto_path, caffe_path)

net = load_net()

# --------- Utilities ---------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    pil_img = ImageOps.exif_transpose(pil_img.convert("RGB"))
    arr = np.array(pil_img)  # RGB
    return arr[:, :, ::-1].copy()  # to BGR

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])

def detect_faces(bgr_img: np.ndarray, conf_threshold: float = 0.6):
    h, w = bgr_img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)),
                                 scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    out = []
    for i in range(detections.shape[2]):
        score = float(detections[0, 0, i, 2])
        if score >= conf_threshold:
            x1, y1, x2, y2 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            out.append((x1, y1, x2, y2, score))
    return out

def draw_boxes(bgr_img: np.ndarray, boxes):
    out = bgr_img.copy()
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out

def handle_image(pil_img: Image.Image, source: str, conf: float, max_width: int):
    bgr = pil_to_bgr(pil_img)
    boxes = detect_faces(bgr, conf_threshold=conf)
    out = draw_boxes(bgr, boxes)
    out_pil = bgr_to_pil(out)

    st.subheader(f"Result — {source}")
    st.image(out_pil, caption=f"Detected {len(boxes)} face(s)", use_container_width=False, width=max_width)

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button("Download result (PNG)", data=buf.getvalue(),
                       file_name="faces_detected.png", mime="image/png")

# --------- UI ---------
st.title("👤 Face Detection ")
st.caption("Upload an image or take a snapshot. Works on Streamlit Cloud (snapshot only, not live video).")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.99, 0.60, 0.01)
    max_width = st.slider("Display width (px)", 400, 1200, 800, 50)
    st.markdown("---")
    st.info("- Use *Camera input* for a snapshot (no live feed on cloud).\n- Ensure good lighting and a frontal face.")

tab1, tab2 = st.tabs(["📤 Upload image", "📷 Camera input"])

with tab1:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        img = Image.open(uploaded)
        handle_image(img, "Uploaded image", conf, max_width)

with tab2:
    snap = st.camera_input("Take a photo")
    if snap is not None:
        img = Image.open(snap)
        handle_image(img, "Camera snapshot", conf, max_width)

st.markdown("---")