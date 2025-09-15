from enum import Enum
from typing import List


class FineTuningFrom(str, Enum):
    IMAGENET = "imagenet"
    RANDOM = "random"
    FUNDUS = "fundus"


class ModelType(str, Enum):
    MULTITASK = "multitask"
    MULTILABEL = "multilabel"
