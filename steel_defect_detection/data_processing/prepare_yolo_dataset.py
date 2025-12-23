"""Prepare YOLO dataset from Severstal Steel Defect Detection data."""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


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


def mask_to_yolo_segment(mask, class_id, img_shape=(256, 1600)):
    """
    Convert binary mask to YOLO segmentation polygon format.

    Args:
        mask: Binary mask (H, W)
        class_id: The class ID for the defect
        img_shape: Original image shape (H, W)

    Returns:
        YOLOv8 segmentation format string (class_id x1 y1 x2 y2 ...)
    """
    H, W = img_shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_labels = []

    for contour in contours:
        if contour.shape[0] < 3:  # Skip contours with less than 3 points
            continue

        # Reshape contour to (N, 2) and normalize coordinates
        segment = contour.reshape(-1, 2)
        normalized_segment = []
        for x, y in segment:
            normalized_segment.append(f"{x / W:.6f}")
            normalized_segment.append(f"{y / H:.6f}")

        yolo_labels.append(f"{class_id}" + " " + " ".join(normalized_segment))

    return "\n".join(yolo_labels)


def prepare_yolo_dataset(
    raw_data_dir: str = "data/raw",
    output_dir: str = "data/yolo_dataset",
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Prepare YOLO segmentation dataset from Severstal data.

    Args:
        raw_data_dir: Directory with raw data (train_images/, train.csv)
        output_dir: Directory to save YOLO dataset
        train_split: Train/val split ratio
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)

    print("=" * 80)
    print("Preparing YOLO Segmentation Dataset")
    print("=" * 80)

    # Create output directories
    for subset in ["train", "val"]:
        (output_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / subset).mkdir(parents=True, exist_ok=True)

    print(f"✓ Created output directories in {output_path}")

    # Load train.csv
    train_csv_path = raw_path / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    print(f"✓ Loaded {len(train_df)} annotations from train.csv")

    # Group by ImageId
    grouped_df = train_df.groupby("ImageId")
    print(f"✓ Found {len(grouped_df)} unique images")

    # Split images into train/val
    unique_image_ids = list(grouped_df.groups.keys())
    random.shuffle(unique_image_ids)

    split_idx = int(len(unique_image_ids) * train_split)
    train_image_ids = unique_image_ids[:split_idx]
    val_image_ids = unique_image_ids[split_idx:]

    print(f"✓ Split: {len(train_image_ids)} train, {len(val_image_ids)} val images")

    # Process images
    IMG_HEIGHT, IMG_WIDTH = 256, 1600
    train_images_dir = raw_path / "train_images"

    train_count = 0
    val_count = 0
    train_ann_count = 0
    val_ann_count = 0

    print("\nProcessing images...")

    for idx, image_id in enumerate(unique_image_ids):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(unique_image_ids)} images...")

        current_image_df = grouped_df.get_group(image_id)

        # Collect YOLO labels for this image
        yolo_labels_for_image = []

        for _, row in current_image_df.iterrows():
            class_id = int(row["ClassId"]) - 1  # YOLO classes are 0-indexed
            encoded_pixels = row["EncodedPixels"]

            if pd.notna(encoded_pixels):
                mask = rle_decode(encoded_pixels, shape=(IMG_HEIGHT, IMG_WIDTH))

                # Convert mask to YOLO format
                yolo_segment_str = mask_to_yolo_segment(
                    mask, class_id, img_shape=(IMG_HEIGHT, IMG_WIDTH)
                )
                if yolo_segment_str:
                    yolo_labels_for_image.append(yolo_segment_str)

        # Determine subset
        if image_id in train_image_ids:
            subset = "train"
            train_count += 1
            train_ann_count += len(yolo_labels_for_image)
        else:
            subset = "val"
            val_count += 1
            val_ann_count += len(yolo_labels_for_image)

        # Copy image
        image_src_path = train_images_dir / image_id
        image_dst_path = output_path / "images" / subset / image_id

        if image_src_path.exists():
            shutil.copy(image_src_path, image_dst_path)
        else:
            print(f"\n  Warning: Image not found: {image_src_path}")
            continue

        # Save YOLO label file
        if yolo_labels_for_image:
            label_filename = image_id.replace(".jpg", ".txt")
            label_path = output_path / "labels" / subset / label_filename

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_labels_for_image))

    print("\n✓ Processed all images")

    # Create data.yaml
    data_yaml = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": ["1", "2", "3", "4"],
    }

    data_yaml_path = output_path / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"✓ Created {data_yaml_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Dataset Preparation Complete!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print(f"Train images: {train_count}")
    print(f"Train annotations: {train_ann_count}")
    print(f"Val images: {val_count}")
    print(f"Val annotations: {val_ann_count}")
    print(f"Data config: {data_yaml_path}")
    print("=" * 80)


if __name__ == "__main__":
    prepare_yolo_dataset()
