"""Data download utilities for Steel Defect Detection dataset."""

import zipfile
from pathlib import Path


def download_data(output_dir: str = "data/raw") -> None:
    """
    Download Severstal Steel Defect Detection dataset from Kaggle.

    Args:
        output_dir: Directory to save downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to {output_path}...")

    # Import Kaggle API only when needed
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download competition files
    competition_name = "severstal-steel-defect-detection"

    try:
        api.competition_download_files(competition_name, path=str(output_path))

        # Extract zip files
        for file in output_path.glob("*.zip"):
            print(f"Extracting {file.name}...")
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(output_path)
            file.unlink()  # Remove zip file after extraction

        print("Dataset downloaded and extracted successfully!")

        # Create train_images and test_images directories if they don't exist
        (output_path / "train_images").mkdir(exist_ok=True)
        (output_path / "test_images").mkdir(exist_ok=True)

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please make sure you have:")
        print("1. Kaggle API credentials configured (~/.kaggle/kaggle.json)")
        print(
            "2. Accepted competition rules at: "
            "https://www.kaggle.com/competitions/severstal-steel-defect-detection"
        )
        raise


if __name__ == "__main__":
    download_data()
