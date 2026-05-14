# 😷 Face Mask Detector

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B)](https://rajneeshbabu-face-mask-detector.streamlit.app)

Real-time 3-class face mask detection in your browser. Upload an image or use your webcam — AI detects faces with an SSD deep learning model and classifies mask usage with a fine-tuned EfficientNetB0 (98.33% accuracy).

🔗 **[Live Demo →](https://rajneeshbabu-face-mask-detector.streamlit.app)**

---

## Classes

| Label | Emoji | Description |
|---|---|---|
| **Mask On** | ✅ | Face mask worn correctly over nose and mouth |
| **Mask Incorrect** | ⚠️ | Mask on chin, neck, or below the nose |
| **No Mask** | ❌ | No face mask present |

---

## How It Works

```
Image (upload or webcam)
         |
         v
OpenCV SSD Face Detector (res10_300x300 Caffe model)
         |
         v
Crop each detected face → resize to 224×224
         |
         v
EfficientNetB0 TFLite FP16 — 3-class classification
         |
         v
Draw bounding box + confidence label (PIL)
         |
         v
Display result + download button (Streamlit)
```

---

## Model Comparison (8-epoch benchmark)

| Model | Val Accuracy | Size | Time |
|---|---|---|---|
| YOLOv8n-cls (TF) | 94.88% | 13.6 MB | 182 s |
| MobileNetV2 | 84.81% | 9.5 MB | 155 s |
| **EfficientNetB0** | **98.33%** | **8.4 MB** | 205 s |

**Winner: EfficientNetB0** — deployed as TFLite FP16 (8.4 MB) via `ai-edge-litert`.

---

## Architecture

```
Input (224×224, RGB, [0–255])
         |
         v
Data Augmentation (flip, rotate, zoom, brightness, contrast)
         |
         v
EfficientNetB0 Backbone — ImageNet pretrained (preprocesses internally)
         |
         v
GlobalAveragePooling2D → BatchNorm → Dropout(0.4)
Dense(256, swish, L2) → Dropout(0.2)
         |
         v
Dense(3, softmax) → [Mask On | Mask Incorrect | No Mask]
```

### Two-Phase Training

| | Phase 1 | Phase 2 |
|---|---|---|
| Goal | Feature extraction | Fine-tuning |
| Backbone | Frozen | Top 15 layers unfrozen |
| LR | 1e-3 (Adam) | 5e-6 (Cosine Decay) |
| Loss | CategoricalCE + label smoothing 0.1 | CategoricalCE + label smoothing 0.05 |
| Class weights | Balanced (sklearn) | Balanced (sklearn) |

---

## Project Structure

```
face-mask-detector/
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── packages.txt                  # apt packages (libgl1, libglib2.0-0t64)
├── runtime.txt                   # Python 3.11 (set in Streamlit App Settings)
├── README.md
├── facemask-training.ipynb       # Full training notebook (run on Kaggle)
├── train_optimized.py            # Standalone training script
├── face_detector/
│   ├── deploy.prototxt           # SSD face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel
└── models/
    ├── face_mask_fp16.tflite     # Deployment model (8.4 MB)  ← used by app.py
    ├── face_mask_efficientnet.keras
    ├── best_model.keras
    └── *.png                     # Training charts
```

---

## Run Locally

### Step 1 — Clone

```bash
git clone https://github.com/rajneeshbabu/face-mask-detector.git
cd face-mask-detector
```

### Step 2 — Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repo + `app.py`
4. **Important:** In App Settings → General → **Python version: 3.11**
   *(Streamlit Cloud currently ignores `runtime.txt` — must be set manually)*
5. Click **Deploy**

---

## Train From Scratch (Kaggle)

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Add dataset: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
3. Upload `facemask-training.ipynb` and click **Run All**
4. Download outputs from `/kaggle/working/` and copy into `models/`

The notebook trains a full two-phase EfficientNetB0 and exports:
- `face_mask_efficientnet.keras` — full Keras model
- `face_mask_fp16.tflite` — quantized deployment model (used by app.py)

---

## Dataset

- **Source**: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
- **Classes**: `with_mask`, `mask_weared_incorrect`, `without_mask`
- **Total**: ~3,800 labeled face images
- **Split**: 80% train / 20% validation (stratified, class-balanced)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with EfficientNetB0 · ai-edge-litert · OpenCV SSD · Streamlit*
