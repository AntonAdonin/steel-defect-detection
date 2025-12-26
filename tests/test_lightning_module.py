"""Simple test script to verify Lightning module works."""

import sys
from pathlib import Path

import torch

from steel_defect_detection.training.efficientnet_module import EfficientNetModule

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



def test_module():
    """Test EfficientNet Lightning Module."""
    print("Testing EfficientNet Lightning Module...")

    # Create module
    model = EfficientNetModule(
        num_classes=4,
        learning_rate=0.001,
        weight_decay=0.0001,
    )

    print("✓ Module created successfully")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)

    assert output.shape == (batch_size, 4), f"Expected shape (4, 4), got {output.shape}"
    print(f"✓ Forward pass works: input {x.shape} -> output {output.shape}")

    # Test training step
    labels = torch.randint(0, 4, (batch_size,))
    batch = (x, labels)
    loss = model.training_step(batch, 0)

    assert loss is not None, "Training step returned None"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"✓ Training step works: loss = {loss.item():.4f}")

    # Test validation step
    loss = model.validation_step(batch, 0)
    assert loss is not None, "Validation step returned None"
    print(f"✓ Validation step works: loss = {loss.item():.4f}")

    # Test optimizer configuration
    optimizers = model.configure_optimizers()
    print(f"✓ Optimizer configured: {type(optimizers)}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_module()
