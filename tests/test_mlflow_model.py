"""Test MLflow model locally before deployment."""

from pathlib import Path

import numpy as np

from steel_defect_detection.inference.mlflow_model import SteelDefectEnsemble


def test_model():
    """Test MLflow pyfunc model."""
    print("=" * 80)
    print("Testing MLflow pyfunc Model")
    print("=" * 80)

    # Check if models exist
    yolo_path = Path("models/yolo_best.pt")
    efficientnet_path = Path("models/efficientnet_steel_defect.pth")

    if not yolo_path.exists():
        print(f"\n❌ YOLO model not found: {yolo_path}")
        print("   Train YOLO first: python commands.py train_yolo")
        return

    if not efficientnet_path.exists():
        print(f"\n❌ EfficientNet model not found: {efficientnet_path}")
        print("   Train EfficientNet first: python commands.py train_efficientnet")
        return

    print(f"\n✓ Found YOLO model: {yolo_path}")
    print(f"✓ Found EfficientNet model: {efficientnet_path}")

    # Create mock context
    class MockContext:
        def __init__(self):
            self.artifacts = {
                "yolo_model": str(yolo_path),
                "efficientnet_model": str(efficientnet_path),
            }

    # Initialize model
    print("\n📦 Loading models...")
    model = SteelDefectEnsemble()
    model.load_context(MockContext())

    print("✓ Models loaded successfully")

    # Create dummy test image
    print("\n🧪 Testing with dummy image...")
    dummy_image = np.random.randint(0, 255, (256, 1600, 3), dtype=np.uint8)

    # Predict
    import pandas as pd

    input_df = pd.DataFrame({"image": [dummy_image]})

    print("   Running prediction...")
    predictions = model.predict(None, input_df)

    print("\n📊 Results:")
    result = predictions[0]
    print(f"   Number of detections: {result['num_detections']}")
    print(f"   Image shape: {result['image_shape']}")

    if result["detections"]:
        print("\n   Detections:")
        for i, det in enumerate(result["detections"], 1):
            print(f"     {i}. BBox: {det['bbox']}")
            print(f"        Confidence: {det['confidence']:.3f}")
            print(f"        Class: {det['class_name']}")

    print("\n" + "=" * 80)
    print("Test Complete! ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Register model: python commands.py register_model")
    print("2. Start MLflow: python commands.py start_mlflow")
    print("3. Start API: python commands.py start_api")
    print("=" * 80)


if __name__ == "__main__":
    test_model()
