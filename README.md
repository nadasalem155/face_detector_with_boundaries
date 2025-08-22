# 👤 Face Detector with Boundaries

A Python program that uses **OpenCV** to detect faces from the webcam and draw **green boundaries** around them 🟩.  
Press **Q** to close the camera.

🌐 **Live Demo on Streamlit:** [face-detector-with-boundaries](https://face-detector-with-boundaries.streamlit.app/)

---

## ✨ Features

- 🧑‍🤝‍🧑 Detect **multiple faces** in an image or snapshot  
- 🎯 Adjustable **confidence threshold** to filter detections  
- 🖼️ Works on both **uploaded images** and **camera snapshots**  
- 💾 Download results in **PNG format**  
- ☁️ Streamlit-based UI — runs on **local machine** or **Streamlit Cloud**  
- 🖥️ Real-time face detection using a **webcam** (local version)  

---

## 💻 Notebook Version

You can also run the project in **Jupyter Notebook**:

- Open `webcam_face_detection.ipynb` in Jupyter.  
- Make sure the model files are in the same folder:
  - `deploy.prototxt`  
  - `res10_300x300_ssd_iter_140000.caffemodel`  
- Run the cells to detect faces from your webcam.  

🔗 **Notebook link:** [webcam_face_detection.ipynb](link-to-your-notebook)

---

## 🚀 Usage

### Streamlit Web App

```bash
streamlit run app.py

Open the URL displayed in the terminal.

Upload an image or use the camera snapshot option.


Local Webcam Version

Run the script:

python webcam_face_detection.py

Press Q to quit the webcam window.



---

📦 Requirements

Python 3.8+ 🐍

OpenCV (opencv-python)

Streamlit (streamlit)

PIL / Pillow (Pillow)

Numpy (numpy)



---

⚠️ Notes

Works on Streamlit Cloud only for snapshots, not live video ☁️

Ensure good lighting and a frontal face for better detection 💡

Confidence threshold can be adjusted for more/less strict detection 🎚️



---
