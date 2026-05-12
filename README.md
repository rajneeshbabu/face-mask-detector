# Face Mask Detector

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

Real-time 3-class face mask detection using your webcam. Detects faces with an SSD deep learning model and classifies mask usage with a fine-tuned EfficientNetB0.

---

## Classes

| Label | Description |
|---|---|
| **Mask On** | Face mask worn correctly over nose and mouth |
| **Mask Worn Incorrectly** | Mask on chin, neck, or below the nose |
| **No Mask** | No face mask present |

---

## How It Works

```
Webcam frame
     |
     v
SSD Face Detector (res10_300x300 Caffe model)
     |
     v
Crop each detected face
     |
     v
EfficientNetB0 TFLite FP16 — 3-class classification
     |
     v
Draw bounding box + label on frame
     |
     v
Live OpenCV window  (press Q to quit)
```

---

## Model Comparison (8-epoch benchmark)

| Model | Val Accuracy | Size | Time |
|---|---|---|---|
| YOLOv8n-cls (TF) | 94.88% | 13.6 MB | 182 s |
| MobileNetV2 | 84.81% | 9.5 MB | 155 s |
| **EfficientNetB0** | **98.33%** | **8.4 MB** | 205 s |

**Winner: EfficientNetB0** — deployed as TFLite FP16 (8.4 MB).

---

## Architecture

```
Input (224x224, RGB, [0-255])
     |
     v
Data Augmentation (flip, rotate, zoom, brightness, contrast)
     |
     v
EfficientNetB0 Backbone — ImageNet pretrained, preprocesses internally
     |
     v
GlobalAveragePooling2D -> BatchNorm -> Dropout(0.4)
Dense(256, swish, L2) -> Dropout(0.2)
     |
     v
Dense(3, softmax) -> [with_mask | mask_weared_incorrect | without_mask]
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
├── app.py                        # main app — run this for real-time detection
├── requirements.txt
├── README.md
├── facemask-training.ipynb       # full training notebook (run on Kaggle)
├── train_optimized.py            # standalone training script
├── face_detector/
│   ├── deploy.prototxt           # SSD face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel   # SSD face detector weights
└── models/
    ├── face_mask_fp16.tflite     # deployment model (8.4 MB)  ← used by app.py
    ├── face_mask_efficientnet.keras
    ├── best_model.keras
    ├── comparison_results.json
    ├── comparison_chart.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── prediction_samples.png
    ├── class_distribution.png
    ├── benchmark_curves.png
    ├── augmentation_preview.png
    └── sample_images.png
```

---

## Setup and Run

### Step 1 — Clone the repo

```bash
git clone https://github.com/rajneeshbabu/face-mask-detector.git
cd face-mask-detector
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
pip install tensorflow            # skip if already installed
```

### Step 4 — Run the detector

```bash
python app.py
```

A window opens showing your webcam feed with live detection. Press **Q** to quit.

---

## Train From Scratch (Kaggle)

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Add the dataset: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
3. Upload `facemask-training.ipynb` and click **Run All**
4. Download the outputs from `/kaggle/working/` — copy them into `models/`

The notebook runs a full two-phase EfficientNetB0 training and exports:
- `face_mask_efficientnet.keras` — full Keras model
- `face_mask_fp16.tflite` — quantized deployment model (used by app.py)

---

## Dataset

- **Source**: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) on Kaggle
- **Classes**: `with_mask`, `mask_weared_incorrect`, `without_mask`
- **Total**: ~3,800 labeled face images
- **Split**: 80% train / 20% validation (stratified, class-balanced)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with TensorFlow · EfficientNetB0 · OpenCV SSD · imutils*
