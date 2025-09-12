# 👤 Face Detection with Boundaries

 A Python program that uses **OpenCV** and a **pre-trained Deep Learning model (ResNet10 SSD)** 
to detect faces from the webcam, draw **green boundaries** 🟩 around them, 
and display the **confidence score (%)** above each detected face.

🌐 **Live Demo on Streamlit:** [face-detector-with-boundaries](https://face-detector-with-boundaries.streamlit.app/)

---

## ✨ Features

- 🧑‍🤝‍🧑 Detect **multiple faces** in an image, snapshot, or webcam feed  
- 🎯 **Adjustable confidence threshold** (slider in sidebar) to control detection strictness  
  - **Higher threshold** → fewer false positives, might miss faint faces  
  - **Lower threshold** → detects more faces but may include non-faces  
- 📊 Displays **number of detected faces** in the result caption  
- 🖼️ Works on both **uploaded images** and **camera snapshots**  
- 💾 Download results in **PNG format** directly from the app  
- 🖥️ Real-time face detection using a **webcam** (local version)  
- ☁️ Streamlit-based UI — works on **local machine** or **Streamlit Cloud**  

---

## 💻 Notebook Version

You can also run the project in **Jupyter Notebook**:

- Open `webcam_face_detection.ipynb` in Jupyter.  
- Make sure the model files are in the same folder:
  - `deploy.prototxt`  
  - `res10_300x300_ssd_iter_140000.caffemodel`  
- Run the cells to detect faces from your webcam.  
- The notebook will show the number of detected faces and their confidence score in real time.

🔗 **Notebook link:** [webcam_face_detection.ipynb](face_detector.ipynb)

---

## 🚀 Usage

### 1️⃣ Streamlit Web App

Run locally with:

streamlit run app.py

Open the URL displayed in the terminal.

📤 Upload an image → App detects faces and shows number of faces detected

📷 Take a snapshot → Works with your camera (snapshot only on Streamlit Cloud)

🎚️ Adjust confidence threshold from the sidebar

💾 Download results as a PNG file



---

### 2️⃣ Local Webcam Version

Run the script:

python webcam_face_detection.py

A window will open with your webcam feed

Green boxes will appear around detected faces

Press Q to quit the webcam window



---

📦 Requirements

Python 3.8+

OpenCV (opencv-python)

Streamlit (streamlit)

Pillow (Pillow)

Numpy (numpy)


Install dependencies with:

pip install -r requirements.txt


---

⚠️ Notes

Works on Streamlit Cloud only for snapshots, not live video ☁️

Ensure good lighting and a frontal face for better detection 💡

Use the confidence slider to tweak detection sensitivity 🎚️
