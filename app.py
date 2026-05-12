"""
Face Mask Detector - Real-Time Webcam
======================================
Face detection  : OpenCV SSD (res10_300x300 Caffe model)
Mask classifier : EfficientNetB0 TFLite FP16 (3-class)
Display         : OpenCV window  |  Press Q to quit
"""

import os
import time
import cv2
import numpy as np
import imutils

# TFLite interpreter
try:
    from tflite_runtime.interpreter import Interpreter
    print("[INFO] Using tflite-runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[INFO] Using TensorFlow TFLite interpreter")


# ── Paths ──────────────────────────────────────────────────────────────────
FACE_PROTO   = "face_detector/deploy.prototxt"
FACE_WEIGHTS = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL   = "models/face_mask_fp16.tflite"

# ── Classes (must match training order) ───────────────────────────────────
CLASS_DISPLAY = ["Mask On", "Mask Incorrect", "No Mask"]
# BGR colors for OpenCV drawing
CLASS_COLORS  = [
    (0, 200, 0),      # green  — mask on
    (0, 165, 255),    # orange — mask incorrect
    (0, 0, 220),      # red    — no mask
]

IMG_SIZE         = 224
FACE_CONFIDENCE  = 0.5    # minimum face detection confidence


# ── Load face detector ─────────────────────────────────────────────────────
print("[INFO] Loading face detector...")
face_net = cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)


# ── Load mask classifier ───────────────────────────────────────────────────
print("[INFO] Loading mask classifier...")
interpreter = Interpreter(model_path=MASK_MODEL)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]


# ── Detect faces and classify mask ─────────────────────────────────────────
def detect_and_predict(frame):
    (h, w) = frame.shape[:2]

    # Build blob for SSD face detector (same mean as reference project)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs  = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < FACE_CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Convert BGR -> RGB, resize to 224x224
        # Keep [0, 255] range — EfficientNetB0 has built-in preprocessing
        face_rgb     = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        faces.append(np.array(face_resized, dtype=np.float32))
        locs.append((x1, y1, x2, y2))

    preds = []
    for face_arr in faces:
        inp = face_arr[np.newaxis]                          # (1, 224, 224, 3)
        interpreter.set_tensor(in_det["index"], inp)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det["index"])[0].astype(np.float32)
        # Softmax normalization
        e     = np.exp(probs - probs.max())
        probs = e / e.sum()
        preds.append(probs)

    return locs, preds


# ── Start webcam ───────────────────────────────────────────────────────────
print("[INFO] Starting webcam... press Q to quit")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        break

    frame = imutils.resize(frame, width=700)
    locs, preds = detect_and_predict(frame)

    for (box, probs) in zip(locs, preds):
        (x1, y1, x2, y2) = box
        pred_idx = int(np.argmax(probs))
        label    = f"{CLASS_DISPLAY[pred_idx]}: {probs[pred_idx]*100:.1f}%"
        color    = CLASS_COLORS[pred_idx]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Mask Detector  |  Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
