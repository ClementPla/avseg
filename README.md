# AVSeg: Artery Vein Segmentation Models in PyTorch

AVSeg provides PyTorch-based models for artery and vein segmentation. It supports both multilabel and multiclass segmentation tasks, with pretrained weights available.

## Features

- **Multilabel & Multiclass segmentation**
- **Pretrained weights** for public datasets
- **Easy integration** with PyTorch workflows

## Installation

```bash
pip install git+https://github.com/ClementPla/avseg.git
```

## Usage

```python
from avseg.segment import segment, ModelType, FineTuningFrom

# Load a test image (replace with your own image path)
img = cv2.imread("test_image.png")[:, :, ::-1]  # BGR to RGB

# Infer:
segmentation = segment(
    img,
    model_type=ModelType.MULTILABEL,
    finetuned_from=FineTuningFrom.IMAGENET,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

## Pretrained Weights

All pretrained weights are hosted on [HuggingFace](https://huggingface.co/ClementP/AVSeg).

They are automatically downloaded and cached when you use the library.

## Dataset

Models are trained on publicly available artery/vein segmentation datasets:
- Aptos: 
- DRHAGIS
- FIVES (train & Test)
- HRF
- IDRID-RETA-train
- INSPIRE
- LES-AV
- MAPLES-DR

## License

MIT
