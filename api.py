from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os
import random
import base64

# --------------------------------------------------
# Load trained model (loaded once at startup)
# --------------------------------------------------
model = tf.keras.models.load_model("traffic_sign_cnn.keras")

# --------------------------------------------------
# Class names (must match training order exactly)
# --------------------------------------------------
CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safe here since this is a demo API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Health check (used by Render + debugging)
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Predict endpoint (used when user drops an image)
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_id = int(np.argmax(preds))

    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": float(preds[class_id])
    }

# --------------------------------------------------
# Random sample images (for Generate button)
# --------------------------------------------------
SAMPLE_DIR = "samples"

@app.get("/random-images")
def random_images(n: int = 5):
    files = os.listdir(SAMPLE_DIR)

    if not files:
        return []

    chosen = random.sample(files, min(n, len(files)))
    results = []

    for fname in chosen:
        with open(os.path.join(SAMPLE_DIR, fname), "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        results.append({
            "image": img_base64
        })

    return results