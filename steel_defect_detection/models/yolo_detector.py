"""YOLO detector wrapper for Steel Defect Detection."""

from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO


class YOLODetector:
    """Wrapper for YOLO model for defect detection."""

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str = "yolo11s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to trained model weights
            model_name: Name of YOLO model to use if no path provided
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run model on
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_name)

        self.model.to(device)

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs/detect",
        name: str = "train",
        **kwargs,
    ):
        """
        Train YOLO model.

        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            project: Project directory
            name: Experiment name
            **kwargs: Additional YOLO training arguments
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=self.device,
            **kwargs,
        )
        return results

    def predict(self, image, save: bool = False, **kwargs):
        """
        Run prediction on image(s).

        Args:
            image: Image path or numpy array
            save: Whether to save prediction results
            **kwargs: Additional prediction arguments

        Returns:
            YOLO results object
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save,
            **kwargs,
        )
        return results

    def detect_boxes(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect bounding boxes in image.

        Args:
            image: Image path or numpy array

        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        results = self.predict(image, save=False)
        boxes = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return boxes

    def export_onnx(self, **kwargs):
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            **kwargs: Additional export arguments
        """
        return self.model.export(format="onnx", **kwargs)
