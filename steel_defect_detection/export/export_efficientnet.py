"""Export EfficientNet model to ONNX format."""

import logging
from pathlib import Path

import hydra
import onnx
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/export", config_name="default")
def export_efficientnet(cfg: DictConfig):
    """
    Export EfficientNet model to ONNX format.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting EfficientNet model export to ONNX...")

    # Load model weights
    model_path = Path(cfg.efficientnet.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Create EfficientNet B0 model from torchvision
    model = models.efficientnet_b0(weights=None)

    # Modify classifier to match num_classes
    num_classes = cfg.efficientnet.num_classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    logger.info(f"Created EfficientNet B0 model with {num_classes} classes")

    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("✓ Model weights loaded successfully")

    # Prepare output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_path = output_dir / "efficientnet.onnx"

    # Create dummy input
    dummy_input = torch.randn(1, 3, cfg.efficientnet.image_size, cfg.efficientnet.image_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        export_params=True,
        opset_version=cfg.efficientnet.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # optional dynamic batch
    )

    logger.info(f"✓ EfficientNet model exported successfully to {export_path}")

    # Verify ONNX model
    if cfg.efficientnet.get("verify", True):
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(str(export_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model verification passed")


if __name__ == "__main__":
    export_efficientnet()
