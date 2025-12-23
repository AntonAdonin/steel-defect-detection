"""Export YOLO model to ONNX format for Triton Inference Server."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/export", config_name="default")
def export_yolo(cfg: DictConfig):
    """
    Export YOLO model to ONNX format.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting YOLO model export to ONNX...")

    # Load model
    model_path = Path(cfg.yolo.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = YOLO(str(model_path))

    # Export to ONNX
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to {output_dir}")
    export_path = model.export(
        format="onnx",
        imgsz=cfg.yolo.image_size,
        dynamic=cfg.yolo.dynamic,
        simplify=cfg.yolo.simplify,
        opset=cfg.yolo.opset_version,
    )

    import shutil

    shutil.copy(export_path, output_dir / "model.onnx")

    logger.info(f"✓ YOLO model exported successfully to {output_dir / 'model.onnx'}")
    logger.info(f"Original export path: {export_path}")


if __name__ == "__main__":
    export_yolo()
