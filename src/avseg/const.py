from enum import Enum
from typing import List
from matplotlib.colors import ListedColormap


class FineTuningFrom(str, Enum):
    IMAGENET = "imagenet"
    RANDOM = "random"
    FUNDUS = "fundus"


class ModelType(str, Enum):
    MULTITASK = "multitask"
    MULTILABEL = "multilabel"


CMAP = ListedColormap(["black", "red", "blue"])
