from functools import lru_cache

import torch
import numpy as np
from avseg.models import BaseModel
from avseg.const import ModelType, FineTuningFrom
from avseg.revision import get_model_revision
from fundus_data_toolkit.functional import (
    autofit_fundus_resolution,
    reverse_autofit_tensor,
)
from fundus_data_toolkit.config import get_normalization
import torchvision.transforms.functional as Ftv
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from typing import Union


@lru_cache(maxsize=2)
def get_model(
    model_type: ModelType, finetuned_from: FineTuningFrom, device, compile=False
) -> BaseModel:
    model = get_model_revision(model_type, finetuned_from)
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model


def segment(
    image: np.ndarray,
    model_type: ModelType,
    finetuned_from: FineTuningFrom,
    image_resolution=1024,
    autofit_resolution=True,
    reverse_autofit=True,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Segment fundus image into either 2 or 3 classes (either multilabel artery/vein or multitask background/artery/vein).

    Args:

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size 5xHxW)
    """

    model = get_model(model_type, finetuned_from, device, compile=compile)
    model.eval()
    h, w, c = image.shape
    if autofit_resolution:
        image, roi, transforms = autofit_fundus_resolution(
            image, image_resolution, return_roi=True
        )

    image = (image / 255.0).astype(np.float32)
    tensor = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to(device)

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]
    tensor = Ftv.normalize(tensor, mean=mean, std=std)
    with torch.inference_mode():
        pred = model(tensor)
        if model_type == ModelType.MULTILABEL:
            pred = torch.sigmoid(pred)
            # We create a background channel by multiplying the artery and vein channels
            background = (1 - pred[:, 0]) * (1 - pred[:, 1])
            pred = torch.cat([background.unsqueeze(1), pred], dim=1)
        else:
            pred = torch.softmax(pred, dim=1)
    pred = pred.squeeze(0).cpu()
    if reverse_autofit and autofit_resolution:
        pred = reverse_autofit_tensor(pred, **transforms)
    return pred


def batch_segment(
    batch: Union[torch.Tensor, np.ndarray],
    model_type: ModelType,
    finetuned_from: FineTuningFrom,
    already_normalized=False,
    mean=None,
    std=None,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Segment batch of fundus images into 5 classes: background, CTW, EX, HE, MA

    Args:


    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size Bx5xHxW)
    """

    model = get_model(model_type, finetuned_from, device, compile=compile)
    model.eval()

    # Check if batch is torch.Tensor or np.ndarray. If np.ndarray, convert to torch.Tensor
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch)  # Convert to torch.Tensor

    batch = batch.to(device)

    # Check if dimensions are BxCxHxW. If not, transpose
    if batch.shape[1] != 3:
        batch = batch.permute((0, 3, 1, 2))

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]

    # Check if batch is normalized. If not, normalize it
    if not already_normalized:
        batch = batch / 255.0
        batch = Ftv.normalize(batch, mean=mean, std=std)

    with torch.inference_mode():
        pred = model(batch)
        if model_type == ModelType.MULTILABEL:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)

    return pred
