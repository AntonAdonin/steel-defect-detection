"""Data download utilities for Steel Defect Detection dataset."""

import urllib.request
import zipfile
from pathlib import Path

KAGGLE_COMPETITION = "severstal-steel-defect-detection"
GDRIVE_URL = "https://drive.google.com/uc?id=1zQ2VKA6ng5M_oXFd6JtM_r6keebwBtfK"
GDRIVE_FILENAME = "severstal-steel-defect-detection.zip"


def _extract_archives(output_path: Path) -> None:
    """Extract all zip archives in directory and remove them."""
    for archive in output_path.glob("*.zip"):
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(output_path)
        archive.unlink()


def _download_from_kaggle(output_path: Path) -> None:
    """Download dataset using Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.competition_download_files(
        KAGGLE_COMPETITION,
        path=str(output_path),
    )


def _download_from_gdrive(output_path: Path) -> None:
    """Download dataset from Google Drive."""
    print("Downloading dataset from Google Drive...")

    destination = output_path / GDRIVE_FILENAME
    urllib.request.urlretrieve(GDRIVE_URL, destination)


def download_data(output_dir: str = "data/raw") -> None:
    """
    Download Severstal Steel Defect Detection dataset.

    Strategy:
    1. Try Kaggle API
    2. Fallback to Google Drive

    Args:
        output_dir: Directory to save downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Preparing dataset in: {output_path.resolve()}")

    try:
        _download_from_kaggle(output_path)
    except Exception as err:
        print(f"Kaggle download failed: {err}")
        print("Falling back to Google Drive...")

        try:
            _download_from_gdrive(output_path)
        except Exception as gdrive_err:
            print("Google Drive download also failed.")
            raise RuntimeError(
                "Failed to download dataset via Kaggle and Google Drive."
            ) from gdrive_err

    _extract_archives(output_path)

    # Create expected directories
    (output_path / "train_images").mkdir(exist_ok=True)
    (output_path / "test_images").mkdir(exist_ok=True)

    print("Dataset downloaded and prepared successfully.")


if __name__ == "__main__":
    download_data()
