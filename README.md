# Face Mask Detector

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-purple)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A 3-class face mask detection system built with **EfficientNetB0** transfer learning. Classifies whether a person is wearing a mask correctly, incorrectly, or not at all.

**Live Demo**: [Deploy on Hugging Face Spaces](#deployment)

---

## Model Comparison Results

Trained on the same dataset for 8 epochs each:

| Model | Params | Size | Val Accuracy | Time |
|---|---|---|---|---|
| YOLOv8n-cls (TF) | 1.11 M | 13.6 MB | 94.88% | 182 s |
| MobileNetV2 | 2.26 M | 9.5 MB | 84.81% | 155 s |
| **EfficientNetB0** | **4.06 M** | **16.9 MB** | **98.33%** | 205 s |

**Winner: EfficientNetB0** — highest accuracy. TFLite FP16 quantization brings it down to **8.4 MB** with less than 0.3% accuracy drop.

---

## Classes

| Label | Description |
|---|---|
| `with_mask` | Face mask worn correctly over nose and mouth |
| `mask_weared_incorrect` | Mask on chin, neck, or below the nose |
| `without_mask` | No face mask present |

---

## Architecture

```
Input (224x224 RGB, [0-255])
     |
     v
Data Augmentation (flip, rotate, zoom, brightness, contrast)
     |
     v
EfficientNetB0 Backbone (ImageNet pretrained, includes preprocessing)
     |
     v
GlobalAveragePooling2D
BatchNormalization
Dropout(0.4)
Dense(256, swish, L2=1e-4)
Dropout(0.2)
     |
     v
Dense(3, softmax) -> [with_mask, mask_weared_incorrect, without_mask]
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
├── facemask_training.ipynb      # full training pipeline (run on Kaggle)
├── app.py                       # Gradio web UI — deploy this
├── requirements.txt
├── README.md
└── models/
    ├── face_mask_fp16.tflite        # deployment model (8.4 MB)  <-- used by app.py
    ├── face_mask_efficientnet.keras # full Keras model (28 MB)
    ├── best_model.keras             # best checkpoint from training
    ├── comparison_results.json      # benchmark numbers
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

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detector.git
cd face-mask-detector
pip install -r requirements.txt
```

### 2. Run the web app locally

```bash
python app.py
# Opens at http://localhost:7860
```

The app loads `models/face_mask_fp16.tflite` automatically.

### 3. Train from scratch (Kaggle)

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Add dataset: `vijaykumar1799/face-mask-detection`
3. Upload `facemask_training.ipynb` and run all cells
4. Download the output files from `/kaggle/working/`

---

## Deployment

### Hugging Face Spaces (free, public URL)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set **SDK = Gradio**
3. Upload these three files:
   - `app.py`
   - `requirements.txt`
   - `face_mask_fp16.tflite`
4. Your app goes live automatically

### Deploy from GitHub to Hugging Face

```bash
# One-time setup: link your GitHub repo to a HF Space
# In your HF Space settings -> Repository -> Link to GitHub repo
# Every push to main will auto-redeploy the Space
```

### Local (with public tunnel)

```bash
# Edit app.py: set share=True in demo.launch()
python app.py
# Gradio prints a public URL valid for 72 hours
```

---

## Dataset

- **Source**: [vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) on Kaggle
- **Classes**: `with_mask`, `mask_weared_incorrect`, `without_mask`
- **Split**: 80% train / 20% validation (stratified)
- **Preprocessing**: Resize to 224x224, raw [0-255] pixel values (EfficientNetB0 normalizes internally)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with TensorFlow · EfficientNetB0 · Gradio*
