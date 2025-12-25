"""Dataset classes for Steel Defect Detection."""

from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class SteelDefectDataset(Dataset):
    """Dataset for Steel Defect Detection with YOLO-style annotations."""

    def __init__(
        self,
        images_dir: str,
        annotations_csv: str | None = None,
        transform=None,
        is_train: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            images_dir: Path to directory with images
            annotations_csv: Path to CSV file with annotations (optional for test set)
            transform: Albumentations transforms
            is_train: Whether this is training dataset
        """
        self.images_dir = Path(images_dir)
        self.is_train = is_train
        self.transform = transform

        # Get list of images
        self.image_files = sorted(self.images_dir.glob("*.jpg"))

        # Load annotations if provided
        self.annotations = None
        if annotations_csv:
            self.annotations = pd.read_csv(annotations_csv)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """Get item by index."""
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.annotations is not None:
            # Get annotations for this image
            img_name = img_path.name
            img_annotations = self.annotations[self.annotations["ImageId"] == img_name]

            # Parse RLE masks and create bounding boxes
            labels = []

            for _, row in img_annotations.iterrows():
                if pd.notna(row["EncodedPixels"]):
                    class_id = int(row["ClassId"]) - 1  # 0-indexed
                    # For now, create dummy boxes - should parse RLE properly
                    labels.append(class_id)

            labels = (
                torch.tensor(labels, dtype=torch.long)
                if labels
                else torch.zeros(0, dtype=torch.long)
            )
        else:
            labels = None

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return {"image": image, "labels": labels, "image_id": img_path.stem}


def get_train_transforms(image_size=(640, 640)):
    """Get training augmentation transforms."""
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size=(640, 640)):
    """Get validation transforms."""
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
