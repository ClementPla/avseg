import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss, GeneralizedDiceLoss
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Dice, JaccardIndex, MetricCollection, F1Score
from avseg.models.pretrained_encoder import get_segmentation_models
from huggingface_hub import PyTorchModelHubMixin


class BaseModel(LightningModule, PyTorchModelHubMixin):
    """
    Base class for all models in the AVSeg project.
    Inherits from PyTorch Lightning's LightningModule.
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.0001,
        arch="unet",
        encoder_name="resnet34",
        pretrained_from="imagenet",
        task="multiclass",
        classes=3,
    ):
        super().__init__()
        if task == "multilabel":
            classes = 2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.arch = arch
        self.encoder_name = encoder_name
        self.pretrained_from = pretrained_from
        self.task = task
        self.model = get_segmentation_models(
            arch=self.arch,
            encoder_name=self.encoder_name,
            num_classes=classes,
            pretrained_on_fundus=self.pretrained_from == "fundus",
        )
        self.av_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            softmax=False,
            sigmoid=True,
        )
        self.vessel_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            softmax=False,
            sigmoid=True,
        )
        # Initialize any common attributes or methods here if needed
        metrics_params = {"task": task, "num_classes": classes, "num_labels": classes}
        self.av_metrics = MetricCollection(
            {
                "AV Dice": F1Score(**metrics_params),
                "AV Jaccard": JaccardIndex(**metrics_params),
                "AV Accuracy": Accuracy(**metrics_params),
            }
        )

        self.vessel_metrics = MetricCollection(
            {
                "Vessel Dice": Dice(),
                "Vessel Jaccard": JaccardIndex(task="binary"),
                "Vessel Accuracy": Accuracy(task="binary"),
            }
        )
        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor.
        :return: Model output.
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        :return: Optimizer instance.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        :param batch: Input batch of data.
        :param batch_idx: Index of the batch.
        :return: Loss value.
        """
        x, y, has_av_labels = batch["image"], batch["mask"], batch["has_av_labels"]
        y.unsqueeze_(1)
        has_av_labels = has_av_labels.bool().squeeze()
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred, y, has_av_labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        :param batch: Input batch of data.
        :param batch_idx: Index of the batch.
        :return: Validation loss value.
        """
        x, y = batch["image"], batch["mask"]
        has_av_labels = batch["has_av_labels"].bool().squeeze()
        y.unsqueeze_(1)
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred, y, has_av_labels)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.compute_metrics(y_pred, y, has_av_labels)
        self.log_dict(
            self.av_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log_dict(
            self.vessel_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def get_loss(self, y_pred, y, has_av_labels, debug=False, show_index=None):
        """
        Calculate the loss for the model.
        :param y_pred: Model predictions.
        :param y: Ground truth labels.
        :return: Loss value.
        """
        if self.task == "multiclass":
            y_pred = y_pred.softmax(dim=1)
            vessel_pred = y_pred[:, 1:].sum(dim=1, keepdim=True)
        else:
            vessel_pred = y_pred.max(dim=1, keepdim=True).values

        y_vessels = (y > 0).long()
        loss_vessel = self.vessel_loss(vessel_pred, y_vessels)
        if has_av_labels.any():
            if self.task == "multiclass":
                y_pred_av = y_pred[has_av_labels]
                y_av = torch.clamp(y[has_av_labels], 0, 2)

                loss_av = self.av_loss(y_pred_av, y_av)
                loss_vessel = loss_vessel * 0.5 + loss_av * 0.5
            else:
                y_with_av = y[has_av_labels].long()
                y_pred_with_av = y_pred[has_av_labels]
                # To one-hot encoding for binary task. We drop the background channel
                y_av = torch.nn.functional.one_hot(
                    y_with_av.clamp(0, 2), num_classes=3
                ).squeeze(1)[:, :, :, 1:]

                y_av[y_with_av.squeeze(1) == 3] = 1  # Convert both artery and vein to 1

                y_av = y_av.permute(0, 3, 1, 2)
                if debug:
                    import matplotlib.pyplot as plt
                    import numpy as np

                    artery = y_av[show_index, 0].cpu().numpy()
                    vein = y_av[show_index, 1].cpu().numpy()

                    combined = [
                        artery[:, :, None],
                        vein[:, :, None],
                        np.zeros_like(artery)[:, :, None],
                    ]
                    combined = np.concatenate(combined, axis=2) * 255

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(artery, cmap="gray")
                    axes[0].set_title("Artery AV Labels")
                    axes[0].axis("off")
                    axes[1].imshow(vein, cmap="gray")
                    axes[1].set_title("Vein AV Labels")
                    axes[1].axis("off")
                    axes[2].imshow(combined.astype(np.uint8))
                    axes[2].set_title("Combined AV Labels")
                    axes[2].axis("off")
                    plt.show()
                loss_av = (
                    self.av_loss(y_pred_with_av[:, :1], y_av[:, :1])
                    + self.av_loss(y_pred_with_av[:, 1:], y_av[:, 1:])
                ) / 2

                loss_vessel = loss_vessel * 0.5 + loss_av * 0.5

        return loss_vessel

    @torch.no_grad()
    def debug_step(self, batch, index):
        x, y, has_av_labels = batch["image"], batch["mask"], batch["has_av_labels"]
        y.unsqueeze_(1)
        has_av_labels = has_av_labels.bool().squeeze()
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred, y, has_av_labels, debug=True, show_index=index)
        self.compute_metrics(y_pred, y, has_av_labels)

    def compute_metrics(self, y_pred, y, has_av_labels):
        """
        Compute metrics for the model.
        :param y_pred: Model predictions.
        :param y: Ground truth labels.
        :param has_av_labels: Boolean tensor indicating presence of AV labels.
        :return: Dictionary of computed metrics.
        """
        if self.task == "multiclass":
            y_pred = y_pred.softmax(dim=1)
            positive_pred = y_pred[:, 1:].sum(dim=1, keepdim=True)
        else:
            y_pred = y_pred.sigmoid()
            positive_pred = y_pred.sum(dim=1, keepdim=True).clamp(0, 1)

        y_vessels = (y > 0).long()

        self.vessel_metrics.update(positive_pred, y_vessels)

        if has_av_labels.any():
            y_pred_av = y_pred[has_av_labels]
            if self.task == "multiclass":
                y_av = torch.clamp(y[has_av_labels], 0, 2)
                y_av = y_av.long()
            else:
                y_av_gt = y[has_av_labels].long()
                # To one-hot encoding for binary task
                y_av = torch.nn.functional.one_hot(
                    y_av_gt.long().clamp(0, 2), num_classes=3
                ).squeeze(1)[:, :, :, 1:]
                y_av[y_av_gt.squeeze(1) == 3] = (
                    1  # Convert crossing to both artery and vein
                )
                y_av = y_av.permute(0, 3, 1, 2)
                # Ensure y_pred_av is in the correct shape
            # Remove channel dimension if present to match prediction shape
            if y_av.dim() == 4 and y_av.shape[1] == 1:
                y_av = y_av.squeeze(1)

            self.av_metrics.update(y_pred_av, y_av)
