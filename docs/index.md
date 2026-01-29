# ObjDet Documentation

Production-grade object detection training framework built on PyTorch Lightning.

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/installation
getting_started/quickstart
getting_started/configuration
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/models
user_guide/data
user_guide/training
user_guide/inference
user_guide/deployment
user_guide/testing
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/models
api/data
api/training
api/inference
api/serving
api/pipelines
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
changelog
```

## Features

- **Multiple Models**: Faster R-CNN, RetinaNet, YOLOv8/v11
- **Flexible Data**: COCO, VOC, YOLO formats with LitData optimization
- **Training**: LightningCLI, custom callbacks, Optuna tuning
- **Deployment**: LitServe REST API with A/B testing
- **MLOps**: Celery job queue, job dependencies, resource routing

## Quick Example

```python
from objdet.models import FasterRCNN
from objdet.data import COCODataModule

# Create model
model = FasterRCNN(num_classes=80)

# Create datamodule
datamodule = COCODataModule(
    data_dir="/data/coco",
    batch_size=16,
)

# Train with Lightning
trainer = Trainer(max_epochs=50)
trainer.fit(model, datamodule)
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
