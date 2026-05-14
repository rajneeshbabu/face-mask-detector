# 😷 Face Mask Detector

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

Real-time 3-class face mask detection — run it in your browser or directly from your webcam. Detects faces with an OpenCV SSD model and classifies mask usage with a fine-tuned EfficientNetB0 (**98.33% accuracy**).

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-mask-detector-hdzvzy2kpppk3cu7m46byj.streamlit.app)

👉 **[Launch Streamlit App →](https://face-mask-detector-hdzvzy2kpppk3cu7m46byj.streamlit.app)**

Upload an image or use your webcam directly in the browser — no installation required.

---

## 🏷️ Classes

| Label | Emoji | Description |
|---|---|---|
| **Mask On** | ✅ | Face mask worn correctly over nose and mouth |
| **Mask Incorrect** | ⚠️ | Mask on chin, neck, or below the nose |
| **No Mask** | ❌ | No face mask present |

---

## 📂 Project Structure

```
face-mask-detector/
├── app.py                        # 🌐 Streamlit web app (upload + browser webcam)
├── app2.py                       # 🎥 Live webcam detector (OpenCV window, local only)
├── requirements.txt              # Python dependencies
├── packages.txt                  # apt packages for Streamlit Cloud
├── runtime.txt                   # Python 3.11 hint for Streamlit Cloud
├── README.md
├── facemask-training.ipynb       # Full training notebook (run on Kaggle)
├── train_optimized.py            # Standalone training script
├── face_detector/
│   ├── deploy.prototxt           # SSD face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel   # SSD weights (~10 MB)
└── models/
    ├── face_mask_fp16.tflite     # Deployment model (8.4 MB) ← used by both apps
    ├── face_mask_efficientnet.keras
    ├── best_model.keras
    └── *.png                     # Training charts
```

---

## 🖥️ App 1 — Streamlit Web App (`app.py`)

**Best for:** sharing, demos, browser-based use, Streamlit Cloud deployment.

- Upload JPG / PNG / WEBP image **or** use your browser webcam (`st.camera_input`)
- Annotated result image with bounding boxes and confidence scores
- Per-face probability bars for all 3 classes
- Download annotated image button
- Works on any device with a browser — no install needed via the Live Demo

### Run locally

```bash
cd "face-mask-detector"
source venv/bin/activate
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎥 App 2 — Live Webcam (`app2.py`)

**Best for:** real-time local detection, continuous webcam feed, offline use.

- Opens a native OpenCV window with live webcam feed
- Processes every frame — detects all faces, classifies each in real time
- FPS counter, face count, per-class counts displayed on screen
- Press **S** to save a screenshot | Press **Q** or **Esc** to quit

### Run locally

```bash
cd "face-mask-detector"
source venv/bin/activate
python app2.py
```

> If your webcam isn't detected, change `VideoCapture(0)` to `VideoCapture(1)` in `app2.py`.

---

## ⚙️ Local Setup (Both Apps)

### Step 1 — Clone the repo

```bash
git clone https://github.com/rajneeshbabu/face-mask-detector.git
cd face-mask-detector
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install "numpy<2.0"
pip install opencv-python-headless Pillow streamlit
```

Install TFLite inference backend:

```bash
# macOS / Windows / Linux (x86_64)
pip install tensorflow

# Linux (for Streamlit Cloud — handled automatically via requirements.txt)
pip install ai-edge-litert
```

### Step 4 — Run

```bash
# Web app
streamlit run app.py

# Live webcam
python app2.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Open **App Settings → General → Python version → 3.11**
   *(Streamlit Cloud ignores `runtime.txt` — must be set manually)*
5. Click **Deploy**

The `packages.txt` and `requirements.txt` handle all server-side dependencies automatically.

---

## 🧠 How It Works

```
Image / Webcam Frame
        │
        ▼
OpenCV SSD Face Detector (res10_300x300 Caffe model)
  → finds bounding boxes for every face in the frame
        │
        ▼
Crop each detected face → resize to 224×224
        │
        ▼
EfficientNetB0 TFLite FP16 — 3-class classification
  → outputs [Mask On, Mask Incorrect, No Mask] probabilities
        │
        ▼
Draw bounding box + label + confidence on image/frame
```

---

## 📊 Model Comparison (8-epoch benchmark)

| Model | Val Accuracy | Size | Train Time |
|---|---|---|---|
| YOLOv8n-cls (TF) | 94.88% | 13.6 MB | 182 s |
| MobileNetV2 | 84.81% | 9.5 MB | 155 s |
| **EfficientNetB0** | **98.33%** | **8.4 MB** | 205 s |

**Winner: EfficientNetB0** — best accuracy and smallest size, deployed as TFLite FP16.

---

## 🏗️ Model Architecture

```
Input (224×224, RGB, [0–255])
        │
        ▼
Data Augmentation
  (horizontal flip, rotation ±15°, zoom ±10%, brightness, contrast)
        │
        ▼
EfficientNetB0 Backbone — ImageNet pretrained
  (preprocesses internally, no manual normalization needed)
        │
        ▼
GlobalAveragePooling2D → BatchNorm → Dropout(0.4)
Dense(256, swish activation, L2 regularization) → Dropout(0.2)
        │
        ▼
Dense(3, softmax) → [Mask On | Mask Incorrect | No Mask]
```

### Two-Phase Training

| | Phase 1 | Phase 2 |
|---|---|---|
| Goal | Feature extraction | Fine-tuning |
| Backbone | Frozen | Top 15 layers unfrozen |
| Learning Rate | 1e-3 (Adam) | 5e-6 (Cosine Decay) |
| Loss | CategoricalCE + label smoothing 0.1 | CategoricalCE + label smoothing 0.05 |
| Class weights | Balanced (sklearn) | Balanced (sklearn) |

---

## 🏋️ Train From Scratch (Kaggle)

1. Go to [kaggle.com](https://kaggle.com) → New notebook
2. Add dataset: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
3. Upload `facemask-training.ipynb` → click **Run All**
4. Download outputs from `/kaggle/working/` → copy into `models/`

The notebook runs full two-phase EfficientNetB0 training and exports:
- `face_mask_efficientnet.keras` — full Keras model
- `face_mask_fp16.tflite` — quantized deployment model (used by both apps)

---

## 📦 Dataset

- **Source**: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) on Kaggle
- **Classes**: `with_mask`, `mask_weared_incorrect`, `without_mask`
- **Total images**: ~3,800 labeled face images
- **Split**: 80% train / 20% validation (stratified, class-balanced)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with EfficientNetB0 · ai-edge-litert · OpenCV SSD · Streamlit*
