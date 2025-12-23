"""CLI commands for Steel Defect Detection project using Fire."""

import subprocess
from pathlib import Path

import fire

from steel_defect_detection.data_processing.download import download_data as _download_data
from steel_defect_detection.data_processing.prepare_classification_dataset import (
    prepare_classification_dataset as _prepare_classification,
)
from steel_defect_detection.data_processing.prepare_yolo_dataset import (
    prepare_yolo_dataset as _prepare_yolo,
)


class Commands:
    """CLI commands for Steel Defect Detection project."""

    def download_data(self, output_dir: str = "data/raw"):
        """
        Download Severstal Steel Defect Detection dataset from Kaggle.

        Args:
            output_dir: Directory to save downloaded data

        Example:
            python commands.py download_data
            python commands.py download_data --output_dir=data/raw
        """
        print("\n Downloading dataset from Kaggle...")
        _download_data(output_dir)
        print(" Download complete!\n")

    def prepare_yolo(
        self,
        raw_data_dir: str = "severstal-steel-defect-detection",
        output_dir: str = "data/yolo_dataset",
        train_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Prepare YOLO segmentation dataset.

        Args:
            raw_data_dir: Directory with raw data
            output_dir: Directory to save YOLO dataset
            train_split: Train/val split ratio
            seed: Random seed

        Example:
            python commands.py prepare_yolo
            python commands.py prepare_yolo --train_split=0.9
        """
        print("\n Preparing YOLO dataset...")
        _prepare_yolo(raw_data_dir, output_dir, train_split, seed)
        print(" YOLO dataset ready!\n")

    def prepare_yolo_kaggle(
        self,
        raw_data_dir: str = "data/raw",
        output_dir: str = "data/yolo_dataset",
        train_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Prepare YOLO segmentation dataset.

        Args:
            raw_data_dir: Directory with raw data
            output_dir: Directory to save YOLO dataset
            train_split: Train/val split ratio
            seed: Random seed

        Example:
            python commands.py prepare_yolo
            python commands.py prepare_yolo --train_split=0.9
        """
        print("\n Preparing YOLO dataset...")
        _prepare_yolo(raw_data_dir, output_dir, train_split, seed)
        print(" YOLO dataset ready!\n")
    
    def prepare_classification(
        self,
        raw_data_dir: str = "severstal-steel-defect-detection",
        output_dir: str = "data/classification_dataset",
        train_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Prepare classification dataset for EfficientNet.

        Args:
            raw_data_dir: Directory with raw data
            output_dir: Directory to save classification dataset
            train_split: Train/val split ratio
            seed: Random seed

        Example:
            python commands.py prepare_classification
            python commands.py prepare_classification --train_split=0.9
        """
        print("\n Preparing classification dataset...")
        _prepare_classification(raw_data_dir, output_dir, train_split, seed)
        print(" Classification dataset ready!\n")

    def prepare_classification_kaggle(
        self,
        raw_data_dir: str = "data/raw",
        output_dir: str = "data/classification_dataset",
        train_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Prepare classification dataset for EfficientNet.

        Args:
            raw_data_dir: Directory with raw data
            output_dir: Directory to save classification dataset
            train_split: Train/val split ratio
            seed: Random seed

        Example:
            python commands.py prepare_classification
            python commands.py prepare_classification --train_split=0.9
        """
        print("\n Preparing classification dataset...")
        _prepare_classification(raw_data_dir, output_dir, train_split, seed)
        print(" Classification dataset ready!\n")

    def prepare_all_data(self, raw_data_dir: str = "data/raw"):
        """
        Prepare both YOLO and classification datasets.

        Args:
            raw_data_dir: Directory with raw data

        Example:
            python commands.py prepare_all_data
        """
        print("\n Preparing all datasets...")
        self.prepare_yolo(raw_data_dir=raw_data_dir)
        self.prepare_classification(raw_data_dir=raw_data_dir)
        print(" All datasets ready!\n")

    def train_yolo(self, config_path: str = None):
        """
        Train YOLO detector with MLflow tracking.

        Args:
            config_path: Path to custom config (optional)

        Example:
            python commands.py train_yolo
            python commands.py train_yolo --config_path=configs/custom.yaml
        """
        print("\n Training YOLO detector...")
        cmd = ["python", "steel_defect_detection/training/train_yolo.py"]
        if config_path:
            cmd.extend(["--config-path", str(Path(config_path).parent)])
            cmd.extend(["--config-name", Path(config_path).stem])

        subprocess.run(cmd, check=True)
        print(" YOLO training complete!\n")

    def train_efficientnet(self, config_path: str = None):
        """
        Train EfficientNet classifier with PyTorch Lightning and MLflow.

        Args:
            config_path: Path to custom config (optional)

        Example:
            python commands.py train_efficientnet
            python commands.py train_efficientnet --config_path=configs/custom.yaml
        """
        print("\n Training EfficientNet classifier...")
        cmd = ["python", "steel_defect_detection/training/train_efficientnet.py"]
        if config_path:
            cmd.extend(["--config-path", str(Path(config_path).parent)])
            cmd.extend(["--config-name", Path(config_path).stem])

        subprocess.run(cmd, check=True)
        print(" EfficientNet training complete!\n")

    def train_all(self):
        """
        Train both YOLO and EfficientNet models.

        Example:
            python commands.py train_all
        """
        print("\n Training all models...")
        self.train_yolo()
        self.train_efficientnet()
        print(" All training complete!\n")

    def export_yolo(self, model_path: str = "models/yolo_best.pt"):
        """
        Export YOLO model to ONNX format.

        Args:
            model_path: Path to YOLO model weights

        Example:
            python commands.py export_yolo
            python commands.py export_yolo --model_path=models/best.pt
        """
        print(f"\n Exporting YOLO model from {model_path}...")
        cmd = ["python", "steel_defect_detection/export/export_yolo.py"]
        subprocess.run(cmd, check=True)
        print(" YOLO export complete!\n")

    def export_efficientnet(self, model_path: str = "models/efficientnet_steel_defect.pth"):
        """
        Export EfficientNet model to ONNX format.

        Args:
            model_path: Path to EfficientNet model weights

        Example:
            python commands.py export_efficientnet
            python commands.py export_efficientnet --model_path=models/my_model.pth
        """
        print(f"\n Exporting EfficientNet model from {model_path}...")
        cmd = ["python", "steel_defect_detection/export/export_efficientnet.py"]
        subprocess.run(cmd, check=True)
        print(" EfficientNet export complete!\n")

    def export_all(self):
        """
        Export both YOLO and EfficientNet models to ONNX.

        Example:
            python commands.py export_all
        """
        print("\n Exporting all models...")
        self.export_yolo()
        self.export_efficientnet()
        print(" All exports complete!\n")

    def start_mlflow(self, host: str = "127.0.0.1", port: int = 8080):
        """
        Start MLflow tracking server.

        Args:
            host: Host address
            port: Port number

        Example:
            python commands.py start_mlflow
            python commands.py start_mlflow --port=5000
        """
        print(f"\n Starting MLflow server on {host}:{port}...")
        print(f"   View at: http://{host}:{port}")
        print("   Press Ctrl+C to stop\n")

        cmd = [
            "mlflow",
            "server",
            "--host",
            host,
            "--port",
            str(port),
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--default-artifact-root",
            "./mlruns",
        ]

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n MLflow server stopped\n")

    def register_model(
        self,
        yolo_path: str = "models/yolo_best.pt",
        efficientnet_path: str = "models/efficientnet_steel_defect.pth",
        model_name: str = "steel_defect_ensemble",
    ):
        """
        Register ensemble model in MLflow Model Registry.

        Args:
            yolo_path: Path to YOLO weights
            efficientnet_path: Path to EfficientNet weights
            model_name: Name for registered model

        Example:
            python commands.py register_model
            python commands.py register_model --model_name=my_model
        """
        print("\n Registering model in MLflow...")

        # Check if models exist
        yolo_file = Path(yolo_path)
        efficientnet_file = Path(efficientnet_path)

        if not yolo_file.exists():
            print(f"\n YOLO model not found: {yolo_path}")
            print("   Train YOLO first: python commands.py train_yolo")
            return

        if not efficientnet_file.exists():
            print(f"\n EfficientNet model not found: {efficientnet_path}")
            print("   Train EfficientNet first: python commands.py train_efficientnet")
            return

        cmd = [
            "python",
            "steel_defect_detection/inference/mlflow_model.py",
            "--yolo_model_path",
            yolo_path,
            "--efficientnet_model_path",
            efficientnet_path,
            "--model_name",
            model_name,
        ]
        subprocess.run(cmd, check=True)
        print(" Model registered!\n")

    def start_api(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Start FastAPI backend server with MLflow Serving.

        Args:
            host: Host address
            port: Port number

        Example:
            python commands.py start_api
            python commands.py start_api --port=8000
        """
        print(f"\n Starting API server on {host}:{port}...")
        print(f"   API docs: http://localhost:{port}/docs")
        print("   Press Ctrl+C to stop\n")

        cmd = [
            "uvicorn",
            "mlflow_api:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
        ]

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n API server stopped\n")

    def lint(self):
        """
        Run code quality checks (black, isort, flake8).

        Example:
            python commands.py lint
        """
        print("\n🔍 Running code quality checks...\n")

        checks = [
            ("Black", ["black", "--check", "."]),
            ("isort", ["isort", "--check-only", "."]),
            ("flake8", ["flake8", ".", "--exclude", ".venv,build,dist"]),
        ]

        failed = []
        for name, cmd in checks:
            print(f"Running {name}...")
            result = subprocess.run(cmd, text=True)
            # Вывод flake8, black, isort сразу в терминал
            if result.returncode != 0:
                failed.append(name)
                print(f"   {name} failed")
            else:
                print(f"   {name} passed")

        if failed:
            print(f"\n Failed checks: {', '.join(failed)}")
            print("Run 'python commands.py format' to auto-fix formatting issues\n")
            return 1
        else:
            print("\n All checks passed!\n")
            return 0

    def format(self):
        """
        Auto-format code with black and isort.

        Example:
            python commands.py format
        """
        print("\n Formatting code...\n")

        formatters = [
            ("Black", ["black", "."]),
            ("isort", ["isort", "."]),
        ]

        for name, cmd in formatters:
            print(f"Running {name}...")
            subprocess.run(cmd, check=True)
            print(f"   {name} complete")

        print("\n Formatting complete!\n")

    def test(self):
        """
        Run tests with pytest.

        Example:
            python commands.py test
        """
        print("\n Running tests...\n")
        cmd = ["pytest", "-v"]
        subprocess.run(cmd)
        print("\n")

    def test_api(self, api_url: str = "http://localhost:8000"):
        """
        Test MLflow API endpoints.

        Args:
            api_url: API base URL

        Example:
            python commands.py test_api
            python commands.py test_api --api_url=http://localhost:8000
        """
        print("\n Testing API...\n")
        cmd = ["python", "tests/test_api.py", "--api_url", api_url]
        subprocess.run(cmd, check=True)
        print("\n")

    def dvc_add_data(self):
        """
        Add data directory to DVC tracking.

        Example:
            python commands.py dvc_add_data
        """
        print("\n Adding data to DVC...\n")
        subprocess.run(["dvc", "add", "data"], check=True)
        print("\n Data added to DVC! Remember to commit .dvc files:\n")
        print("  git add data.dvc .gitignore")
        print("  git commit -m 'Add data to DVC'\n")

    def dvc_pull(self):
        """
        Pull data from DVC remote storage.

        Example:
            python commands.py dvc_pull
        """
        print("\n Pulling data from DVC...\n")
        subprocess.run(["dvc", "pull"], check=True)
        print("\n Data pulled successfully!\n")

    def dvc_push(self):
        """
        Push data to DVC remote storage.

        Example:
            python commands.py dvc_push
        """
        print("\n Pushing data to DVC...\n")
        subprocess.run(["dvc", "push"], check=True)
        print("\n Data pushed successfully!\n")

    def dvc_status(self):
        """
        Check DVC status.

        Example:
            python commands.py dvc_status
        """
        print("\n DVC Status:\n")
        subprocess.run(["dvc", "status"], check=True)
        print("\n")


def main():
    """Main entry point for Fire CLI."""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
