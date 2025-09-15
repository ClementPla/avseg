from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from nntools.utils import Config
from avseg.models import BaseModel
from avseg.data import BaseDataModule
from pathlib import Path
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.tuner import Tuner
import argparse
import torch

torch.set_float32_matmul_precision("medium")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVSeg Training Script")
    parser.add_argument(
        "--pretrained_from",
        type=str,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Pretrained encoder to use for the model",
    )
    args = parser.parse_args()

    config = Config("configs/config.yaml")

    config["model"]["pretrained_from"] = (
        args.pretrained_from if args.pretrained_from != "None" else None
    )
    config["model"]["encoder_name"] = args.encoder

    data_module = BaseDataModule(**config["data"])
    data_module.setup("fit")
    model = BaseModel(**config["model"], **config["training"])
    logger = WandbLogger(project="AVSeg", config=config.tracked_params)

    try:
        run_name = logger.experiment.name
        path = Path("checkpoints") / "AVSeg" / run_name
    except TypeError:
        path = Path("checkpoints") / "AVSeg" / "default"
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    callbacks = [
        ModelCheckpoint(
            dirpath=path,
            monitor="Vessel Jaccard",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="Vessel Jaccard",
            mode="max",
            patience=5,
            verbose=True,
        ),
    ]

    trainer = Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=logger,
    )
    tuner = Tuner(trainer=trainer)

    lr_finder = tuner.lr_find(
        model=model,
        max_lr=0.05,
        min_lr=1e-6,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        early_stop_threshold=None,
        num_training=100,
    )
    new_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {new_lr}")
    config["training"]["learning_rate"] = new_lr
    model.learning_rate = new_lr
    model.hparams.learning_rate = new_lr

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
