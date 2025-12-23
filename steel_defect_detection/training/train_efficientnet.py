"""Training script for EfficientNet classifier using PyTorch Lightning and MLflow."""

import subprocess
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from steel_defect_detection.data_processing.classification_datamodule import (
    SteelDefectClassificationDataModule,
)
from steel_defect_detection.training.efficientnet_module import EfficientNetModule


def get_git_commit_id():
    """Get current git commit ID."""
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit_id
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train_efficientnet(cfg: DictConfig):
    """
    Train EfficientNet classifier with PyTorch Lightning.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("EfficientNet Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Initialize DataModule
    print("\n[1/5] Initializing DataModule...")
    datamodule = SteelDefectClassificationDataModule(
        data_dir=cfg.data.classification_dataset_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.model.efficientnet.image_size,
        pin_memory=cfg.data.pin_memory,
    )

    # Setup data to get num_classes
    datamodule.prepare_data()
    datamodule.setup("fit")
    num_classes = datamodule.num_classes

    # Initialize Model
    print("\n[2/5] Initializing EfficientNet Model...")
    model = EfficientNetModule(
        num_classes=num_classes,
        learning_rate=cfg.train.efficientnet.learning_rate,
        weight_decay=cfg.train.efficientnet.weight_decay,
        optimizer=cfg.train.efficientnet.optimizer,
        scheduler=cfg.train.efficientnet.scheduler,
        scheduler_patience=cfg.train.efficientnet.scheduler_patience,
        scheduler_factor=cfg.train.efficientnet.scheduler_factor,
        pretrained=cfg.model.efficientnet.pretrained,
    )

    # Setup MLflow Logger
    print("\n[3/5] Setting up MLflow Logger...")
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=f"efficientnet_{cfg.model.efficientnet.name}",
    )

    # Log git commit ID
    git_commit = get_git_commit_id()
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "git_commit_id", git_commit)

    # Log configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow_logger.log_hyperparams(config_dict)

    # Setup Callbacks
    print("\n[4/5] Setting up Callbacks...")
    checkpoint_dir = Path(cfg.train.checkpoint.dirpath) / "efficientnet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="efficientnet-{epoch:02d}-{val/loss:.4f}-{val/f1:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.train.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        RichProgressBar(),
    ]

    # Initialize Trainer
    print("\n[5/5] Initializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.train.efficientnet.epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        precision=cfg.train.trainer.precision,
        log_every_n_steps=cfg.train.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.trainer.check_val_every_n_epoch,
        logger=mlflow_logger,
        callbacks=callbacks,
        deterministic=True,
        enable_checkpointing=cfg.train.trainer.enable_checkpointing,
        enable_progress_bar=cfg.train.trainer.enable_progress_bar,
        enable_model_summary=cfg.train.trainer.enable_model_summary,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    trainer.fit(model, datamodule=datamodule)

    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)

    # Save best checkpoint path
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model checkpoint: {best_model_path}")

    # Save model weights in PyTorch format
    final_model_path = Path(cfg.paths.models_dir) / "efficientnet_steel_defect.pth"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), final_model_path)
    print(f"Model weights saved to: {final_model_path}")

    # Log model artifact to MLflow
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(final_model_path))

    # Test on validation set with best model
    print("\n" + "=" * 80)
    print("Testing best model on validation set...")
    print("=" * 80)
    trainer.test(model, datamodule=datamodule, ckpt_path=best_model_path)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model: {best_model_path}")
    print(f"MLflow run ID: {mlflow_logger.run_id}")
    print(f"View results: {cfg.mlflow.tracking_uri}")


if __name__ == "__main__":
    train_efficientnet()
