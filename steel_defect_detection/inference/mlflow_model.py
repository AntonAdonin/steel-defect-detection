"""MLflow pyfunc wrapper for Steel Defect Detection ensemble (YOLO + EfficientNet)."""

import base64
import io
from pathlib import Path
from typing import Dict

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


class SteelDefectEnsemble(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc model for Steel Defect Detection ensemble.

    Uses YOLO for detection and EfficientNet for classification.
    """

    def load_context(self, context):
        """
        Load models from MLflow artifacts.

        Args:
            context: MLflow context with artifacts
        """
        # Load YOLO model
        yolo_path = context.artifacts["yolo_model"]
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.conf = 0.25  # Confidence threshold
        self.yolo_model.iou = 0.45  # IoU threshold

        # Load EfficientNet model
        efficientnet_path = context.artifacts["efficientnet_model"]

        # Import here to avoid issues
        from torchvision.models import efficientnet_b0

        self.efficientnet_model = efficientnet_b0(weights=None)

        # Replace classifier
        num_classes = 4
        num_ftrs = self.efficientnet_model.classifier[1].in_features
        self.efficientnet_model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

        # Load weights
        state_dict = torch.load(efficientnet_path, map_location="cpu")
        self.efficientnet_model.load_state_dict(state_dict)
        self.efficientnet_model.eval()

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.efficientnet_model.to(self.device)

        # EfficientNet transforms
        self.efficientnet_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(f"✓ Models loaded on device: {self.device}")

    def predict(self, context, model_input):
        """
        Run ensemble prediction on input images.

        Args:
            context: MLflow context
            model_input: Input data (pandas DataFrame with 'image' column or dict)

        Returns:
            List of predictions with bounding boxes and classes
        """
        # Handle different input formats
        if hasattr(model_input, "to_dict"):
            # DataFrame
            images = model_input["image"].tolist()
        elif isinstance(model_input, dict):
            images = model_input.get("image", model_input.get("images", []))
        elif isinstance(model_input, list):
            images = model_input
        else:
            raise ValueError(f"Unsupported input type: {type(model_input)}")

        results = []

        for img_data in images:
            result = self._predict_single(img_data)
            results.append(result)

        return results

    def _predict_single(self, img_data) -> Dict:
        """
        Predict on a single image.

        Args:
            img_data: Image data (path, numpy array, or base64 string)

        Returns:
            Dictionary with detections and classifications
        """
        # Load image
        image = self._load_image(img_data)

        # Step 1: YOLO detection
        yolo_results = self.yolo_model(image, verbose=False)[0]

        detections = []

        if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            for box in yolo_results.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])

                # Crop detected region
                crop = image[y1:y2, x1:x2]

                # Step 2: EfficientNet classification
                defect_class = self._classify_crop(crop)

                detection = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class": int(defect_class),
                    "class_name": f"defect_{defect_class + 1}",
                }

                detections.append(detection)

        return {
            "num_detections": len(detections),
            "detections": detections,
            "image_shape": list(image.shape[:2]),
        }

    def _load_image(self, img_data) -> np.ndarray:
        """
        Load image from various formats.

        Args:
            img_data: Image data (path, numpy array, or base64)

        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        if isinstance(img_data, str):
            if img_data.startswith("data:image") or img_data.startswith("/9j"):
                # Base64 encoded image
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                # File path
                return cv2.imread(img_data)

        elif isinstance(img_data, np.ndarray):
            return img_data

        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        else:
            raise ValueError(f"Unsupported image format: {type(img_data)}")

    def _classify_crop(self, crop: np.ndarray) -> int:
        """
        Classify cropped defect region with EfficientNet.

        Args:
            crop: Cropped image region (BGR)

        Returns:
            Class ID (0-3)
        """
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # Apply transforms
        input_tensor = self.efficientnet_transform(crop_pil).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.efficientnet_model(input_tensor)
            pred_class = logits.argmax(dim=1).item()

        return pred_class


def register_model(
    yolo_model_path: str = "models/yolo_best.pt",
    efficientnet_model_path: str = "models/efficientnet_steel_defect.pth",
    model_name: str = "steel_defect_ensemble",
    tracking_uri: str = "http://127.0.0.1:8080",
):
    """
    Register ensemble model in MLflow Model Registry.

    Args:
        yolo_model_path: Path to YOLO weights
        efficientnet_model_path: Path to EfficientNet weights
        model_name: Name for MLflow model
        tracking_uri: MLflow tracking server URI
    """
    print("=" * 80)
    print("Registering Steel Defect Ensemble in MLflow Model Registry")
    print("=" * 80)

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Validate model files exist
    yolo_path = Path(yolo_model_path)
    efficientnet_path = Path(efficientnet_model_path)

    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    if not efficientnet_path.exists():
        raise FileNotFoundError(f"EfficientNet model not found: {efficientnet_path}")

    print(f"\n✓ Found YOLO model: {yolo_path}")
    print(f"✓ Found EfficientNet model: {efficientnet_path}")

    # Define artifacts
    artifacts = {
        "yolo_model": str(yolo_path),
        "efficientnet_model": str(efficientnet_path),
    }

    # Define conda environment
    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "mlflow",
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "ultralytics>=8.0.0",
                    "opencv-python>=4.8.0",
                    "pillow>=10.0.0",
                    "numpy<2.0.0",
                ]
            },
        ],
        "name": "steel_defect_env",
    }

    # Example input
    import pandas as pd

    input_example = pd.DataFrame(
        {"image": ["path/to/image.jpg"]}  # Example: can be path, numpy array, or base64
    )

    # Log model
    print("\n📦 Logging model to MLflow...")
    with mlflow.start_run(run_name="ensemble_registration") as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SteelDefectEnsemble(),
            artifacts=artifacts,
            conda_env=conda_env,
            input_example=input_example,
            signature=mlflow.models.infer_signature(
                input_example,
                [
                    {
                        "num_detections": 2,
                        "detections": [{"bbox": [0, 0, 100, 100], "confidence": 0.95, "class": 0}],
                    }
                ],
            ),
        )

        print(f"✓ Model logged: {model_info.model_uri}")

        # Register model
        print(f"\n📝 Registering model as '{model_name}'...")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)

        print(f"✓ Model registered: {registered_model.name} v{registered_model.version}")

        # Transition to Staging
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=registered_model.version, stage="Staging"
        )

        print("✓ Model transitioned to 'Staging' stage")

    print("\n" + "=" * 80)
    print("Model Registration Complete!")
    print("=" * 80)
    print(f"Model name: {model_name}")
    print(f"Version: {registered_model.version}")
    print("Stage: Staging")
    print("\nLoad model with:")
    print(f'  model = mlflow.pyfunc.load_model("models:/{model_name}/Staging")')
    print("=" * 80)


if __name__ == "__main__":
    import fire

    fire.Fire(register_model)
