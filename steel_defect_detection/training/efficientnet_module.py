"""PyTorch Lightning Module for EfficientNet Steel Defect Classifier."""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision import models


class EfficientNetModule(LightningModule):
    """Lightning Module for EfficientNet classifier."""

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer: str = "Adam",
        scheduler: str = "ReduceLROnPlateau",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        pretrained: bool = True,
    ):
        """
        Initialize EfficientNet Lightning Module.

        Args:
            num_classes: Number of defect classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            optimizer: Optimizer type (Adam, SGD, AdamW)
            scheduler: Learning rate scheduler type
            scheduler_patience: Patience for ReduceLROnPlateau
            scheduler_factor: Factor for ReduceLROnPlateau
            pretrained: Use pretrained ImageNet weights
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        if pretrained:
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = models.efficientnet_b0(weights=None)

        # Replace classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Metrics storage
        self.training_step_outputs: list = []
        self.validation_step_outputs: list = []

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # Store for epoch-level metrics
        self.training_step_outputs.append(
            {"loss": loss.detach(), "preds": preds.detach(), "targets": labels.detach()}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

        # Store for epoch-level metrics
        self.validation_step_outputs.append(
            {"loss": loss.detach(), "preds": preds.detach(), "targets": labels.detach()}
        )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)

        return {"preds": preds.detach(), "targets": labels.detach()}

    def on_train_epoch_end(self):
        """Calculate epoch-level training metrics."""
        if not self.training_step_outputs:
            return

        # Aggregate predictions and targets
        all_preds = torch.cat([x["preds"] for x in self.training_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.training_step_outputs])

        # Move to CPU for sklearn metrics
        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()

        # Calculate metrics
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)

        # Log metrics
        self.log("train/f1", f1, prog_bar=True)
        self.log("train/precision", precision)
        self.log("train/recall", recall)

        # Clear memory
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Calculate epoch-level validation metrics."""
        if not self.validation_step_outputs:
            return

        # Aggregate predictions and targets
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        # Move to CPU for sklearn metrics
        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()

        # Calculate metrics
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)

        # Log metrics
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/precision", precision)
        self.log("val/recall", recall)

        # Clear memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Scheduler
        if self.hparams.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
            }
        elif self.hparams.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        images, _ = batch
        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        return preds
