"""PyTorch Lightning DataModule for EfficientNet Classification."""

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class SteelDefectClassificationDataModule(pl.LightningDataModule):
    """DataModule for Steel Defect Classification with EfficientNet."""

    def __init__(
        self,
        data_dir: str = "data/classification_dataset",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        pin_memory: bool = True,
    ):
        """
        Initialize Classification DataModule.

        Args:
            data_dir: Root directory with train/val subdirectories
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            image_size: Target image size for EfficientNet (224 for B0)
            pin_memory: Whether to pin memory in dataloaders
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        # Define transforms
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        Download and prepare data if needed.

        This method is called only on 1 GPU/process in distributed training.
        """
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Please run data preparation script first."
            )

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"Train/Val directories not found in {self.data_dir}\n"
                "Expected structure:\n"
                "  {data_dir}/train/class_1/...\n"
                "  {data_dir}/train/class_2/...\n"
                "  {data_dir}/val/class_1/...\n"
                "  {data_dir}/val/class_2/..."
            )

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / "val"

            self.train_dataset = ImageFolder(train_dir, transform=self.train_transforms)
            self.val_dataset = ImageFolder(val_dir, transform=self.val_transforms)

            print(f"Train dataset: {len(self.train_dataset)} images")
            print(f"Val dataset: {len(self.val_dataset)} images")
            print(f"Number of classes: {len(self.train_dataset.classes)}")
            print(f"Classes: {self.train_dataset.classes}")

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        """Create prediction dataloader."""
        return self.val_dataloader()

    @property
    def num_classes(self):
        """Get number of classes."""
        if self.train_dataset is not None:
            return len(self.train_dataset.classes)
        return 4  # Default for steel defect dataset
