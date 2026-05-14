"""
Face Mask Detector — Streamlit Web App
=======================================
Face detection  : OpenCV SSD (res10_300x300 Caffe model)
Mask classifier : EfficientNetB0 TFLite FP16 (3-class)
Modes           : Upload image  |  Live webcam (st.camera_input)
"""

import os
import io
import numpy as np
import streamlit as st
import cv2
from PIL import Image, ImageDraw, ImageFont

# ── TFLite interpreter ────────────────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter      # ai-edge-litert (Google, py3.9-3.12)
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter  # tflite-runtime fallback
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter                # TF full fallback
        except ImportError:
            Interpreter = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/rajneeshbabu/face-mask-detector",
        "Report a bug": "https://github.com/rajneeshbabu/face-mask-detector/issues",
        "About":        "**Face Mask Detector** — EfficientNetB0 3-class, 98.33% accuracy\nBuilt by [Rajneesh](https://github.com/rajneeshbabu)",
    }
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  header[data-testid="stHeader"]{
    background:rgba(5,5,15,0.9)!important;
    backdrop-filter:blur(8px);
    border-bottom:1px solid rgba(99,102,241,.2)
  }
  .stApp { background: #050b18; }
  .main-title {
    font-size: 2.6rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { text-align:center; color:#94a3b8; font-size:1rem; margin-bottom:1.5rem; }
  .result-card {
    background: linear-gradient(135deg, #1e1b4b, #0f172a);
    border: 1px solid #6366f1; border-radius: 16px;
    padding: 1.5rem; text-align: center; margin: 0.5rem 0;
  }
  .label-mask    { font-size:2rem; font-weight:900; color:#22c55e; }
  .label-wrong   { font-size:2rem; font-weight:900; color:#f59e0b; }
  .label-nomask  { font-size:2rem; font-weight:900; color:#ef4444; }
  .confidence    { font-size:1.1rem; color:#94a3b8; margin-top:0.3rem; }
  .stat-card {
    background:#0f172a; border:1px solid #334155;
    border-radius:12px; padding:0.8rem; text-align:center;
  }
  .stat-val { font-size:1.4rem; font-weight:800;
    background:linear-gradient(135deg,#6366f1,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .stat-lbl { font-size:0.7rem; color:#64748b; margin-top:0.2rem; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(".")
FACE_PROTO   = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
FACE_WEIGHTS = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL   = os.path.join(BASE_DIR, "models", "face_mask_fp16.tflite")

CLASS_NAMES  = ["Mask On", "Mask Incorrect", "No Mask"]
CLASS_COLORS = [(34, 197, 94), (245, 158, 11), (239, 68, 68)]   # RGB: green, amber, red
CLASS_EMOJI  = ["✅", "⚠️", "❌"]
IMG_SIZE     = 224
FACE_CONF    = 0.5


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_face_detector():
    if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_WEIGHTS):
        return None
    return cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)


@st.cache_resource(show_spinner=False)
def load_mask_model():
    if Interpreter is None or not os.path.exists(MASK_MODEL):
        return None, None, None
    interp = Interpreter(model_path=MASK_MODEL)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    return interp, in_det, out_det


# ── Inference ─────────────────────────────────────────────────────────────────
def detect_and_predict(img_rgb: np.ndarray, face_net, interp, in_det, out_det,
                       conf_threshold: float = FACE_CONF):
    """
    img_rgb : uint8 RGB numpy array
    Returns : list of (x1,y1,x2,y2, class_idx, confidence, probs)
    """
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Face detection
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    dets = face_net.forward()

    results = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        inp = face_resized[np.newaxis]
        interp.set_tensor(in_det["index"], inp)
        interp.invoke()
        probs = interp.get_tensor(out_det["index"])[0].astype(np.float32)
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

        cls_idx = int(np.argmax(probs))
        results.append((x1, y1, x2, y2, cls_idx, float(probs[cls_idx]), probs))

    return results


def draw_results(img_rgb: np.ndarray, results: list) -> Image.Image:
    """Draw bounding boxes + labels on a PIL image."""
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    draw    = ImageDraw.Draw(img_pil)

    for (x1, y1, x2, y2, cls_idx, conf, probs) in results:
        color = CLASS_COLORS[cls_idx]
        label = f"{CLASS_NAMES[cls_idx]}  {conf*100:.1f}%"

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        text_w = len(label) * 7
        text_h = 20
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w, y1], fill=color)
        draw.text((x1 + 3, y1 - text_h - 2), label, fill=(255, 255, 255))

    return img_pil


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 😷 Face Mask Detector")
    st.markdown("---")

    face_net               = load_face_detector()
    interp, in_det, out_det = load_mask_model()

    # Model status
    st.markdown("**Model Status:**")
    st.markdown(f"{'✅' if face_net  else '❌'} Face Detector (SSD)")
    st.markdown(f"{'✅' if interp    else '❌'} Mask Classifier (TFLite)")
    st.markdown("---")

    # Detection settings
    conf_thresh = st.slider("Face Confidence Threshold", 0.3, 0.95, FACE_CONF, 0.05)
    st.markdown("---")

    st.markdown("**3 Classes:**")
    st.markdown("✅ &nbsp; Mask On")
    st.markdown("⚠️ &nbsp; Mask Incorrect")
    st.markdown("❌ &nbsp; No Mask")
    st.markdown("---")

    st.markdown("**Model:**")
    st.markdown("EfficientNetB0 TFLite FP16")
    st.markdown("**Accuracy:** 98.33%")
    st.markdown("**Dataset:** RMFD + MAFA")
    st.markdown("---")
    st.caption("Built by [Rajneesh](https://github.com/rajneeshbabu) · MIT License")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">😷 Face Mask Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an image or use your webcam — AI detects faces and classifies mask usage in real time</div>',
    unsafe_allow_html=True,
)

if face_net is None or interp is None:
    st.error("Model files not found. Make sure `face_detector/` and `models/face_mask_fp16.tflite` exist.")
    st.stop()

# ── Input tabs ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂 Upload Image", "📷 Webcam"])

img_rgb = None
source  = None

with tab1:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"],
        help="Best results with clear face images"
    )
    if uploaded:
        img_rgb = np.array(Image.open(uploaded).convert("RGB"))
        source  = "upload"

with tab2:
    st.markdown("Click **Take Photo** to capture from your webcam.")
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        img_rgb = np.array(Image.open(cam_img).convert("RGB"))
        source  = "webcam"

# ── Process ───────────────────────────────────────────────────────────────────
if img_rgb is not None:
    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🖼️ Detection Result")
        with st.spinner("Detecting faces and classifying masks..."):
            results = detect_and_predict(img_rgb, face_net, interp, in_det, out_det,
                                         conf_threshold=conf_thresh)
            annotated = draw_results(img_rgb, results)

        st.image(annotated, use_container_width=True)

        # Stats row
        n_faces   = len(results)
        n_mask    = sum(1 for r in results if r[4] == 0)
        n_wrong   = sum(1 for r in results if r[4] == 1)
        n_nomask  = sum(1 for r in results if r[4] == 2)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, n_faces,  "Faces"),
            (c2, n_mask,   "With Mask"),
            (c3, n_wrong,  "Incorrect"),
            (c4, n_nomask, "No Mask"),
        ]:
            col.markdown(
                f"<div class='stat-card'><div class='stat-val'>{val}</div>"
                f"<div class='stat-lbl'>{lbl}</div></div>",
                unsafe_allow_html=True
            )

    with col2:
        st.subheader("📊 Results")

        if not results:
            st.info("No faces detected. Try a clearer image or lower the confidence threshold in the sidebar.")
        else:
            for i, (x1, y1, x2, y2, cls_idx, conf, probs) in enumerate(results):
                label_class = ["label-mask", "label-wrong", "label-nomask"][cls_idx]
                st.markdown(f"""
                <div class="result-card">
                    <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.3rem">Face {i+1}</div>
                    <div class="{label_class}">{CLASS_EMOJI[cls_idx]} {CLASS_NAMES[cls_idx]}</div>
                    <div class="confidence">Confidence: {conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                for j, (cname, prob) in enumerate(zip(CLASS_NAMES, probs)):
                    c_l, c_r = st.columns([3, 1])
                    c_l.markdown(f"{CLASS_EMOJI[j]} **{cname}**")
                    c_r.markdown(f"**{prob*100:.1f}%**")
                    bar_color = ["#22c55e", "#f59e0b", "#ef4444"][j]
                    st.markdown(
                        f"<div style='background:#1e293b;border-radius:4px;height:8px;margin-bottom:8px'>"
                        f"<div style='background:{bar_color};width:{prob*100:.1f}%;height:8px;border-radius:4px'></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                if i < len(results) - 1:
                    st.markdown("---")

        # Download annotated image
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Result",
            data=buf.getvalue(),
            file_name="face_mask_result.png",
            mime="image/png",
            use_container_width=True
        )

else:
    # Landing
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem">
        <div style="font-size:5rem;margin-bottom:1rem">😷</div>
        <h3 style="color:#94a3b8">Upload an image or use your webcam to get started</h3>
        <p style="color:#64748b">Supports JPG · PNG · WEBP · Live webcam capture</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🧠", "EfficientNetB0", "TFLite FP16 — 98.33% accuracy\nFast inference on CPU"),
        (c2, "🔍", "SSD Face Detector", "OpenCV res10 Caffe model\nDetects multiple faces"),
        (c3, "🏷️", "3 Classes", "Mask On · Mask Incorrect\nNo Mask"),
    ]:
        col.markdown(f"""
        <div class="stat-card" style="padding:1.2rem">
            <div style="font-size:2.2rem">{icon}</div>
            <div style="color:#a5b4fc;font-weight:700;margin:.5rem 0;font-size:.95rem">{title}</div>
            <div style="color:#64748b;font-size:.82rem;white-space:pre-line;line-height:1.5">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
