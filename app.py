"""
=============================================================================
FACE MASK DETECTOR - GRADIO WEB APP  (3-Class)
=============================================================================
Classes:
  0 - with_mask              (green)
  1 - mask_weared_incorrect  (orange)
  2 - without_mask           (red)

Model priority: TFLite FP16 -> native Keras (.keras)

Usage (local):
    pip install -r requirements.txt
    python app.py

Usage (Hugging Face Spaces):
    Upload app.py, requirements.txt, face_mask_fp16.tflite
    Set SDK = "gradio" in Space settings
=============================================================================
"""

import os
import numpy as np
from PIL import Image

# Try lightweight TFLite runtime first, fall back to full TensorFlow
try:
    from tflite_runtime.interpreter import Interpreter
    print("[INFO] Using tflite-runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[INFO] Using full TensorFlow interpreter")

import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASS_LABELS  = ["with_mask", "mask_weared_incorrect", "without_mask"]
CLASS_DISPLAY = ["Mask On", "Mask Worn Incorrectly", "No Mask"]
CLASS_COLORS  = ["#27AE60", "#F39C12", "#E74C3C"]   # green, orange, red
CLASS_ICONS   = ["Mask On", "Mask Incorrect", "No Mask"]
IMG_SIZE = 224

# Model search order — files are inside the models/ folder
MODEL_PATHS = [
    ("models/face_mask_fp16.tflite",        "TFLite FP16"),
    ("models/face_mask_efficientnet.keras", "Keras native"),
    ("models/best_model.keras",             "Keras checkpoint"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    for path, name in MODEL_PATHS:
        if not os.path.exists(path):
            continue
        print(f"[INFO] Loading {name} from {path}")
        if path.endswith(".tflite"):
            interp = Interpreter(model_path=path)
            interp.allocate_tensors()
            in_det  = interp.get_input_details()[0]
            out_det = interp.get_output_details()[0]
            return "tflite", interp, in_det, out_det, name
        else:
            import tensorflow as tf
            m = tf.keras.models.load_model(path, compile=False)
            return "keras", m, None, None, name
    raise FileNotFoundError(
        "No model file found. Expected one of:\n" +
        "\n".join(f"  {p}" for p, _ in MODEL_PATHS)
    )


model_type, model_obj, in_details, out_details, model_name = load_model()


# ---------------------------------------------------------------------------
# Preprocessing — keep [0, 255] range; EfficientNetB0 preprocesses internally
# ---------------------------------------------------------------------------
def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)   # [0, 255]
    return arr[np.newaxis]                   # shape (1, 224, 224, 3)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(pil_image: Image.Image):
    if pil_image is None:
        return (
            "<p style='color:gray;text-align:center;padding:20px;'>"
            "Upload an image to get a prediction.</p>",
            None, ""
        )

    inp = preprocess(pil_image)

    if model_type == "tflite":
        if in_details["dtype"] == np.uint8:
            scale, zp = in_details["quantization"]
            inp = (inp / scale + zp).astype(np.uint8)
        model_obj.set_tensor(in_details["index"], inp)
        model_obj.invoke()
        probs = model_obj.get_tensor(out_details["index"])[0].astype(np.float32)
        if out_details["dtype"] == np.uint8:
            sc, zp = out_details["quantization"]
            probs = (probs.astype(np.float32) - zp) * sc
    else:
        probs = model_obj.predict(inp, verbose=0)[0]

    # Softmax normalization (numerical safety)
    e = np.exp(probs - probs.max())
    probs = e / e.sum()

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    label      = CLASS_DISPLAY[pred_idx]
    color      = CLASS_COLORS[pred_idx]

    result_html = f"""
    <div style="text-align:center; padding:20px; border-radius:14px;
                background:{color}22; border:2px solid {color}; margin:8px 0;">
        <h2 style="color:{color}; margin:8px 0;">{label}</h2>
        <p style="font-size:1.2em; margin:4px 0; color:#333;">
            Confidence: <b>{confidence:.1f}%</b>
        </p>
    </div>
    """

    label_probs = {CLASS_DISPLAY[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}

    debug = f"Model: {model_name} | " + " | ".join(
        f"{CLASS_LABELS[i]}: {probs[i]*100:.1f}%" for i in range(len(CLASS_LABELS))
    )

    return result_html, label_probs, debug


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
TITLE = "Face Mask Detector"
DESCRIPTION = """
Upload a face photo to classify mask usage into one of three categories.

**Model**: EfficientNetB0 (transfer learning, ImageNet pretrained)
**Dataset**: ~3,800 labeled face images | **3-class classification**
**Benchmark accuracy**: 98.33% (8.4 MB TFLite FP16)
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            img_input  = gr.Image(type="pil", label="Upload Face Image", image_mode="RGB")
            submit_btn = gr.Button("Detect", variant="primary")
            gr.Markdown("Supports: JPG, PNG, WEBP")

        with gr.Column(scale=1):
            result_html = gr.HTML(label="Result")
            confidence  = gr.Label(label="Class Probabilities", num_top_classes=3)
            debug_text  = gr.Textbox(label="Debug Info", interactive=False, visible=False)

    submit_btn.click(fn=predict, inputs=[img_input], outputs=[result_html, confidence, debug_text])
    img_input.change(fn=predict,  inputs=[img_input], outputs=[result_html, confidence, debug_text])

    gr.Markdown("""
---
**Classes**
- **Mask On** — face mask worn correctly over nose and mouth
- **Mask Worn Incorrectly** — mask on chin, neck, or below the nose
- **No Mask** — no face mask present
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
