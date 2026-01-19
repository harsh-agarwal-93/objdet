# Quickstart

This guide will get you running object detection training in minutes.

## Training a Model

### Using the CLI

The easiest way to train is using the CLI with a config file:

```bash
# Train Faster R-CNN on COCO
objdet fit --config configs/coco_frcnn.yaml
```

### Using Python

```python
import lightning as L
from objdet.models import FasterRCNN
from objdet.data import COCODataModule

# Initialize model
model = FasterRCNN(
    num_classes=80,
    backbone="resnet50_fpn_v2",
    pretrained_backbone=True,
)

# Initialize data
datamodule = COCODataModule(
    data_dir="/path/to/coco",
    batch_size=8,
)

# Train
trainer = L.Trainer(
    max_epochs=50,
    accelerator="cuda",
    devices=1,
)
trainer.fit(model, datamodule)
```

## Running Inference

```python
from objdet.inference import Predictor

# Load trained model
predictor = Predictor.from_checkpoint("outputs/best.ckpt")

# Run inference
result = predictor.predict("image.jpg")

print(f"Found {len(result['boxes'])} objects")
for box, label, score in zip(
    result["boxes"], result["labels"], result["scores"]
):
    print(f"  {label}: {score:.2f} at {box}")
```

## Deploying as REST API

```python
from objdet.serving import run_server

run_server(
    checkpoint_path="outputs/best.ckpt",
    host="0.0.0.0",
    port=8000,
)
```

Then send requests:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

## Next Steps

- See [Configuration](configuration.md) for detailed config options
- Explore [Models](../user_guide/models.md) for available architectures
- Learn about [Data Formats](../user_guide/data.md) for dataset preparation
