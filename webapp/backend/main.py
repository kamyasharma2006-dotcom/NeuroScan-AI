import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import File, HTTPException, UploadFile
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from fastapi import FastAPI

# ==========================================
# STEP 1: CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="NeuroScan AI",
    description="Brain MRI Classification API",
    version="1.0"
)

# ==========================================
# ALLOW LOCAL REACT FRONTEND TO ACCESS API
# ==========================================

import os

FRONTEND_URL = os.getenv("FRONTEND_URL", "").rstrip("/")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

# ==========================================
# STEP 2: DEFINE MODEL PATH
# ==========================================

# main.py is inside webapp/backend
# Go up two levels to reach the project root.

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "cnn_v2_trained.keras"

# ==========================================
# STEP 3: LOAD TRAINED CNN MODEL
# ==========================================

print("Loading CNN V2 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("CNN V2 model loaded successfully!")

# ==========================================
# STEP 4: CREATE HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "application": "NeuroScan AI",
        "message": "Backend is running successfully!",
        "model": "CNN V2",
        "status": "ready"
    }

# ==========================================
# STEP 5: CREATE HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape)
    }

# ==========================================
# STEP 6: MRI CLASSIFICATION ENDPOINT
# ==========================================

CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):

    # Check the uploaded file type
    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG MRI image."
        )

    # Read the uploaded image
    contents = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    # Decode and preprocess the image
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Unsupported image format")

            image = image.convert("L")

            # Match training preprocessing
            image = image.resize(
                (128, 128),
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image."
        )

    # CNN input shape: (1, 128, 128, 1)
    input_array = image_array[np.newaxis, :, :, np.newaxis]

    # Run the trained CNN
    predictions = model.predict(
        input_array,
        verbose=0
    )

    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    # Return results to the frontend
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            name: round(float(prob) * 100, 2)
            for name, prob in zip(CLASS_NAMES, probabilities)
        },
        "disclaimer": (
            "Research and educational use only. "
            "Not intended for medical diagnosis."
        )
    }
