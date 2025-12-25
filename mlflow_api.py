"""FastAPI backend for Steel Defect Detection using MLflow Serving.

Refactored: uses Lifespan (async context manager), improved style and helpers,
and stores model in app.state.model instead of a global variable.
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow_api")

# Configuration (can be overridden with env vars)
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "steel_defect_ensemble")
MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Staging")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")


# ==== Pydantic models ====
class Detection(BaseModel):
    """Single detection result."""

    bbox: List[int]
    confidence: float
    class_id: int = 0
    class_name: str


class PredictionResponse(BaseModel):
    """Prediction response."""

    num_detections: int
    detections: List[Detection]
    image_shape: List[int]
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    model_stage: str
    timestamp: str


# ==== Helpers ====
def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """Convert PIL.Image (RGB) to numpy array (H, W, 3)."""
    return np.array(image)


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 string to numpy RGB image array."""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    try:
        raw = base64.b64decode(image_base64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return pil_to_numpy_rgb(img)


def numpy_from_upload_file(upload_file: UploadFile) -> np.ndarray:
    """Read UploadFile to numpy RGB array (synchronous helper used in async endpoint)."""
    contents = upload_file.file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    return pil_to_numpy_rgb(pil_img)


def parse_model_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model result dict to expected keys."""
    # Ensure keys exist and normalize class id/name fields that differ by implementations
    detections = []
    for det in result.get("detections", []):
        # support both 'class' and 'class_id'
        class_id = det.get("class") if "class" in det else det.get("class_id", 0)
        class_name = det.get("class_name") or f"defect_{int(class_id) + 1}"
        detections.append(
            {
                "bbox": det.get("bbox", []),
                "confidence": float(det.get("confidence", 0.0)),
                "class_id": int(class_id),
                "class_name": class_name,
            }
        )

    return {
        "num_detections": int(result.get("num_detections", len(detections))),
        "detections": detections,
        "image_shape": result.get("image_shape", []),
    }


# ==== Lifespan (load/unload model) ====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context that loads MLflow model into app.state.model."""
    logger.info("Lifespan starting - attempting to load MLflow model.")
    app.state.model = None
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info("Loading model from MLflow: %s", model_uri)
        try:
            app.state.model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✓ Model loaded: %s", model_uri)
        except Exception as err:
            logger.error("Failed to load model from MLflow: %s", err)
            logger.warning("API will run without model. Register model to enable prediction.")
    except Exception as err:
        logger.exception("Unexpected error during MLflow setup: %s", err)

    try:
        yield
    finally:
        # If needed, perform any cleanup here.
        logger.info("Lifespan stopping - cleaning up model state.")
        app.state.model = None


# ==== App initialization ====
app = FastAPI(
    title="Steel Defect Detection API",
    description="Detect and classify steel defects via MLflow pyfunc model",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==== Routes ====
@app.get("/", response_model=dict)
async def root() -> Dict[str, Any]:
    """Root endpoint with basic service info."""
    return {
        "name": "Steel Defect Detection API",
        "version": app.version,
        "status": "running",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "endpoints": {
            "health": "/health",
            "predict_image": "/predict/image",
            "predict_base64": "/predict/base64",
            "model_info": "/model/info",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    loaded = getattr(app.state, "model", None) is not None
    return HealthResponse(
        status="healthy" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        timestamp=datetime.utcnow().isoformat(),
    )


async def _run_prediction_on_numpy(image_np: np.ndarray) -> Dict[str, Any]:
    """Run model.predict on a numpy RGB image and normalize output."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise RuntimeError("Model not loaded")

    import pandas as pd  # local import to keep module light if not used

    # model expects 'image' column as in the pyfunc wrapper
    input_df = pd.DataFrame({"image": [image_np]})
    raw_preds = model.predict(input_df)
    if not raw_preds or not isinstance(raw_preds, (list, tuple)):
        raise RuntimeError("Model returned unexpected prediction format")
    parsed = parse_model_result(raw_preds[0])
    return parsed


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict on uploaded image file."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = datetime.utcnow()
    try:
        # UploadFile.file.read() is synchronous; it's acceptable here because small test files.
        image_np = numpy_from_upload_file(file)
        parsed = await _run_prediction_on_numpy(image_np)

        detections = [
            Detection(
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"],
            )
            for det in parsed["detections"]
        ]

        processing_time_ms = (datetime.utcnow() - start).total_seconds() * 1000.0
        return PredictionResponse(
            num_detections=parsed["num_detections"],
            detections=detections,
            image_shape=parsed["image_shape"],
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - keep generic for API error handling
        logger.exception("Prediction error (/predict/image): %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class Base64Request(BaseModel):
    """Request model for base64 image payload."""

    image_base64: str


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64Request) -> PredictionResponse:
    """Predict on base64-encoded image string."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = datetime.utcnow()
    try:
        image_np = decode_base64_image(request.image_base64)
        parsed = await _run_prediction_on_numpy(image_np)

        detections = [
            Detection(
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"],
            )
            for det in parsed["detections"]
        ]

        processing_time_ms = (datetime.utcnow() - start).total_seconds() * 1000.0
        return PredictionResponse(
            num_detections=parsed["num_detections"],
            detections=detections,
            image_shape=parsed["image_shape"],
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time_ms,
        )
    except ValueError as ve:
        logger.warning("Bad request to /predict/base64: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction error (/predict/base64): %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_uri": f"models:/{MODEL_NAME}/{MODEL_STAGE}",
        "loaded": True,
    }


# Allow running uvicorn directly on this module for dev convenience
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mlflow_api:app", host="0.0.0.0", port=8000, reload=True)
