import huggingface_hub
import timm
from segmentation_models_pytorch import create_model
from avseg.models.MSMA.model import MSMA

COLLECTION = "ClementP/fundus-grading-665e582701ca1c80a0b5797a"


def name_conversion(name):
    return f"tu-{name}"


def get_models_in_collections(collection_name):
    return list(huggingface_hub.get_collection(collection_name).items)


def convert_collection_item_to_encoder(item):
    model_id = item.item_id
    encoder = model_id.split("-")[1]
    return (encoder, item)


list_of_models = get_models_in_collections(COLLECTION)
list_of_encoders = [convert_collection_item_to_encoder(item) for item in list_of_models]


def get_pretrained_model(encoder_name):
    """
    Get a pretrained model from the collection based on the encoder name.
    :param encoder_name: Name of the encoder to retrieve.
    :return: Pretrained model.
    """
    for item in list_of_encoders:
        if item[0] == encoder_name:
            return item
    raise ValueError(f"Encoder {encoder_name} not found in the collection.")


def get_segmentation_models(
    arch, encoder_name, num_classes=3, pretrained_on_fundus=False
):
    """
    Get a segmentation model based on the encoder name.
    :param encoder_name: Name of the encoder to retrieve.
    :return: Segmentation model.
    """
    if arch == "MSMA":
        return MSMA(in_channels=3, num_classes=num_classes)

    item = get_pretrained_model(encoder_name)
    encoder_smp = name_conversion(item[0])
    model = create_model(
        arch=arch,
        encoder_name=encoder_smp,
        classes=num_classes,
    )
    if pretrained_on_fundus:
        timm_encoder = timm.create_model(
            f"hf-hub:{item[1].item_id}",
            pretrained=True,
            num_classes=1,
        )
        timm_state_dict = timm_encoder.state_dict()
        if "convnext" in encoder_smp:
            # Replace the first . by _ in the key names
            timm_state_dict = {
                k.replace(".", "_", 1): v for k, v in timm_state_dict.items()
            }

        incompatible_keys = model.encoder.model.load_state_dict(
            timm_state_dict, strict=False
        )

    return model
