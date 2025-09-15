from avseg.const import ModelType, FineTuningFrom
from avseg.models import BaseModel


def get_model_revision(
    model_type: ModelType, fine_tuning_from: FineTuningFrom
) -> BaseModel:
    """Get the revision (version) of the model based on its type and fine-tuning source.

    Args:
        model_type (ModelType): Type of the model (MULTITASK or MULTILABEL).
        fine_tuning_from (FineTuningFrom): Source of fine-tuning (IMAGENET, RANDOM, or FUNDUS).

    Returns:
        str: The revision string corresponding to the model.
    """

    if model_type == ModelType.MULTILABEL:
        encoder_name = "seresnext50_32x4d"

    elif model_type == ModelType.MULTITASK:
        encoder_name = "seresnet50"
        if fine_tuning_from == FineTuningFrom.RANDOM:
            raise ValueError(
                "RANDOM initialization is not available for MULTITASK models."
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    revision = f"{model_type.value}-{encoder_name}-{fine_tuning_from.value}"

    return BaseModel.from_pretrained("ClementP/AVSeg", revision=revision)
