"""Training script for YOLO detector using Ultralytics and MLflow."""

import subprocess
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO


def get_git_commit_id():
    """Get current git commit ID."""
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit_id
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train_yolo(cfg: DictConfig):
    """
    Train YOLO detector with MLflow tracking.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("YOLO Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup MLflow
    print("\n[1/5] Setting up MLflow...")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=f"yolo_{cfg.model.yolo.model_name}") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Log git commit ID
        git_commit = get_git_commit_id()
        mlflow.log_param("git_commit_id", git_commit)

        # Log configuration parameters
        print("\n[2/5] Logging hyperparameters...")
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Log YOLO specific params
        yolo_params = {
            "model_name": cfg.model.yolo.model_name,
            "epochs": cfg.train.yolo.epochs,
            "imgsz": cfg.train.yolo.imgsz,
            "batch": cfg.train.yolo.batch,
            "optimizer": cfg.train.yolo.optimizer,
            "lr0": cfg.train.yolo.lr0,
            "lrf": cfg.train.yolo.lrf,
            "momentum": cfg.train.yolo.momentum,
            "weight_decay": cfg.train.yolo.weight_decay,
            "warmup_epochs": cfg.train.yolo.warmup_epochs,
            "patience": cfg.train.yolo.patience,
            "seed": cfg.seed,
        }
        mlflow.log_params(yolo_params)

        # Initialize YOLO model
        print("\n[3/5] Initializing YOLO model...")
        model_name = cfg.model.yolo.model_name
        if cfg.model.yolo.pretrained:
            model = YOLO(f"{model_name}.pt")
            print(f" Loaded pretrained {model_name}")
        else:
            model = YOLO(f"{model_name}.yaml")
            print(f" Created {model_name} from scratch")

        # Prepare data.yaml path
        data_yaml = Path(cfg.data.raw_data_dir) / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"Data configuration not found: {data_yaml}\n"
                "Please prepare YOLO dataset with data.yaml first."
            )

        # Train the model
        print("\n[4/5] Starting training...")
        print("=" * 80)

        results = model.train(
            data=str(data_yaml),
            epochs=cfg.train.yolo.epochs,
            imgsz=cfg.train.yolo.imgsz,
            batch=cfg.train.yolo.batch,
            device=cfg.device,
            optimizer=cfg.train.yolo.optimizer,
            lr0=cfg.train.yolo.lr0,
            lrf=cfg.train.yolo.lrf,
            momentum=cfg.train.yolo.momentum,
            weight_decay=cfg.train.yolo.weight_decay,
            warmup_epochs=cfg.train.yolo.warmup_epochs,
            patience=cfg.train.yolo.patience,
            save_period=cfg.train.yolo.save_period,
            project=cfg.train.yolo.project,
            name=cfg.train.yolo.name,
            exist_ok=True,
            pretrained=cfg.model.yolo.pretrained,
            seed=cfg.seed,
        )

        print("=" * 80)
        print("Training complete!")

        # Log metrics
        print("\n[5/5] Logging metrics and artifacts...")

        # Get the last training results
        results_dir = Path(cfg.train.yolo.project) / cfg.train.yolo.name

        # Log final metrics
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"final/{key}", value)

        # Log training plots
        plots_to_log = [
            "results.png",
            "confusion_matrix.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "PR_curve.png",
        ]

        for plot_name in plots_to_log:
            plot_path = results_dir / plot_name
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), "plots")
                print(f" Logged {plot_name}")

        # Log best model weights
        best_weights = results_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), "models")
            print(" Logged best model weights")

            # Copy to models directory
            models_dir = Path(cfg.paths.models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            target_path = models_dir / "yolo_best.pt"

            import shutil

            shutil.copy(best_weights, target_path)
            print(f" Copied best weights to {target_path}")

        # Log last model weights
        last_weights = results_dir / "weights" / "last.pt"
        if last_weights.exists():
            mlflow.log_artifact(str(last_weights), "models")
            print(" Logged last model weights")

        # Log results.csv if exists
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), "metrics")

            # Parse and log per-epoch metrics
            import pandas as pd

            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()  # Remove whitespace from column names

            for idx, row in df.iterrows():
                epoch = int(row["epoch"]) if "epoch" in row else idx
                for col in df.columns:
                    if col != "epoch":
                        value = row[col]
                        if pd.notna(value) and isinstance(value, (int, float)):
                            mlflow.log_metric(col.strip(), float(value), step=epoch)

            print(" Logged per-epoch metrics from results.csv")

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best model: {best_weights}")
        print(f"Results directory: {results_dir}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"View results: {cfg.mlflow.tracking_uri}")
        print("=" * 80)


if __name__ == "__main__":
    train_yolo()
