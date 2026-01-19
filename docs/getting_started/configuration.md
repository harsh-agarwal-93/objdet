# Configuration

ObjDet uses LightningCLI for configuration management. This allows
you to configure all aspects of training via YAML files.

## Configuration Structure

A complete configuration includes:

```yaml
# Model configuration
model:
  class_path: objdet.models.FasterRCNN
  init_args:
    num_classes: 80
    backbone: resnet50_fpn_v2

# Data configuration
data:
  class_path: objdet.data.COCODataModule
  init_args:
    data_dir: /path/to/coco
    batch_size: 16

# Trainer configuration
trainer:
  max_epochs: 50
  accelerator: cuda
  devices: 1
  precision: 16-mixed
```

## Overriding Config via CLI

```bash
# Override specific values
objdet fit --config base.yaml \
  --model.init_args.num_classes=10 \
  --data.init_args.batch_size=32

# Multiple config files (merged)
objdet fit --config base.yaml --config experiment.yaml
```

## Available Models

| Model | Class Path | Notes |
|-------|------------|-------|
| Faster R-CNN | `objdet.models.FasterRCNN` | Two-stage detector |
| RetinaNet | `objdet.models.RetinaNet` | Single-stage, focal loss |
| YOLOv8 | `objdet.models.YOLOv8` | Ultralytics wrapper |
| YOLOv11 | `objdet.models.YOLOv11` | Latest YOLO |

## Available DataModules

| Format | Class Path | Annotation |
|--------|------------|------------|
| COCO | `objdet.data.COCODataModule` | JSON |
| VOC | `objdet.data.VOCDataModule` | XML |
| YOLO | `objdet.data.YOLODataModule` | TXT |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | None |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name | objdet |
| `CELERY_BROKER_URL` | RabbitMQ URL | amqp://localhost |
