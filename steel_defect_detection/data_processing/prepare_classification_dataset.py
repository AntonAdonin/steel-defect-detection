"""Prepare classification dataset for EfficientNet from cropped defects."""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decode run-length encoded mask.

    Args:
        mask_rle: Run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends, strict=False):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def prepare_classification_dataset(
    raw_data_dir: str = "data/raw",
    output_dir: str = "data/classification_dataset",
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Prepare classification dataset by cropping defects from images.

    Args:
        raw_data_dir: Directory with raw data (train_images/, train.csv)
        output_dir: Directory to save classification dataset
        train_split: Train/val split ratio
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)

    print("=" * 80)
    print("Preparing Classification Dataset")
    print("=" * 80)

    # Create output directories
    for subset in ["train", "val"]:
        for class_id in ["1", "2", "3", "4"]:
            (output_path / subset / class_id).mkdir(parents=True, exist_ok=True)

    print(f" Created output directories in {output_path}")

    # Load train.csv
    train_csv_path = raw_path / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    print(f" Loaded {len(train_df)} annotations from train.csv")

    # Split data by image IDs to avoid data leakage
    unique_image_ids = train_df["ImageId"].unique()
    random.shuffle(unique_image_ids)

    split_idx = int(len(unique_image_ids) * train_split)
    train_image_ids = set(unique_image_ids[:split_idx])

    print(
        f" Split: {len(train_image_ids)} train images, "
        f"{len(unique_image_ids) - len(train_image_ids)} val images"
    )

    # Process images and create crops
    train_images_dir = raw_path / "train_images"
    train_count = {subset: {str(i): 0 for i in range(1, 5)} for subset in ["train", "val"]}

    print("\nProcessing images and creating crops...")

    for idx, row in train_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)} annotations...")

        image_id = row["ImageId"]
        class_id = str(row["ClassId"])
        encoded_pixels = row["EncodedPixels"]

        # Skip if no defect
        if pd.isna(encoded_pixels):
            continue

        # Determine subset
        subset = "train" if image_id in train_image_ids else "val"

        # Load original image
        image_path = train_images_dir / image_id
        if not image_path.exists():
            print(f"\n  Warning: Image not found: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"\n  Warning: Could not load image {image_path}: {e}")
            continue

        # Decode RLE mask
        mask = rle_decode(encoded_pixels, shape=(256, 1600))

        # Find bounding box from mask
        rows, cols = np.where(mask == 1)
        if len(rows) == 0 or len(cols) == 0:
            continue

        min_y, max_y = np.min(rows), np.max(rows)
        min_x, max_x = np.min(cols), np.max(cols)

        # Crop image
        cropped_img = img.crop((min_x, min_y, max_x + 1, max_y + 1))

        # Save cropped image
        base_filename = os.path.splitext(image_id)[0]
        crop_filename = f"{base_filename}_class{class_id}_{idx}.jpg"
        save_path = output_path / subset / class_id / crop_filename

        cropped_img.save(save_path)
        train_count[subset][class_id] += 1

    print("\n Processed all annotations")

    # Summary
    print("\n" + "=" * 80)
    print("Dataset Preparation Complete!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print("\nTrain set:")
    for class_id in ["1", "2", "3", "4"]:
        count = train_count["train"][class_id]
        print(f"  Class {class_id}: {count} images")
    print(f"  Total: {sum(train_count['train'].values())} images")

    print("\nValidation set:")
    for class_id in ["1", "2", "3", "4"]:
        count = train_count["val"][class_id]
        print(f"  Class {class_id}: {count} images")
    print(f"  Total: {sum(train_count['val'].values())} images")

    print("=" * 80)


if __name__ == "__main__":
    prepare_classification_dataset()
