"""PyTorch Lightning DataModule for Steel Defect Detection."""

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from steel_defect_detection.data_processing.dataset import (
    SteelDefectDataset,
    get_train_transforms,
    get_val_transforms,
)


class SteelDefectDataModule(pl.LightningDataModule):
    """DataModule for Steel Defect Detection."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 16,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        image_size=(640, 640),
        pin_memory: bool = True,
    ):
        """
        Initialize DataModule.

        Args:
            data_dir: Root directory with data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_val_split: Train/validation split ratio
            image_size: Target image size (height, width)
            pin_memory: Whether to pin memory in dataloaders
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.image_size = image_size
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download data if needed."""
        # Check if data exists
        if not (self.data_dir / "train_images").exists():
            print("Data not found. Please run download_data() first.")

    def setup(self, stage=None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            # Full training dataset
            full_dataset = SteelDefectDataset(
                images_dir=self.data_dir / "train_images",
                annotations_csv=self.data_dir / "train.csv",
                transform=get_train_transforms(self.image_size),
                is_train=True,
            )

            # Split into train and validation
            train_size = int(len(full_dataset) * self.train_val_split)
            val_size = len(full_dataset) - train_size

            self.train_dataset, val_dataset_temp = random_split(
                full_dataset, [train_size, val_size]
            )

            # Create validation dataset with different transforms
            self.val_dataset = SteelDefectDataset(
                images_dir=self.data_dir / "train_images",
                annotations_csv=self.data_dir / "train.csv",
                transform=get_val_transforms(self.image_size),
                is_train=True,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SteelDefectDataset(
                images_dir=self.data_dir / "test_images",
                annotations_csv=None,
                transform=get_val_transforms(self.image_size),
                is_train=False,
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
