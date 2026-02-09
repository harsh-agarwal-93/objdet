# Models

`objdet` supports a variety of object detection architectures, ranging from accurate two-stage detectors to fast single-stage models.

## Torchvision Models

These models are based on [PyTorch Vision](https://pytorch.org/vision/stable/models.html) implementations. They are generally robust and integrated tightly with PyTorch.

### Faster R-CNN

Two-stage detector known for high accuracy.

**Configuration:**
```yaml
model:
  class_path: objdet.models.torchvision.FasterRCNN
  init_args:
    num_classes: 80
    backbone: resnet50_fpn_v2
    pretrained: false
    pretrained_backbone: true
    trainable_backbone_layers: 3
    min_size: 800
    max_size: 1333
    learning_rate: 0.001
    weight_decay: 0.0001
    optimizer: adamw
    scheduler: cosine
```

### RetinaNet

Single-stage detector with Focal Loss, balancing speed and accuracy.

**Configuration:**
```yaml
model:
  class_path: objdet.models.torchvision.RetinaNet
  init_args:
    num_classes: 80
    backbone: resnet50_fpn_v2
    pretrained: false
    pretrained_backbone: true
    learning_rate: 0.001
    weight_decay: 0.0001
```

## YOLO Models

We support YOLO models via the [Ultralytics](https://github.com/ultralytics/ultralytics) library. These are state-of-the-art for speed/accuracy trade-offs.

### YOLOv8

**Configuration:**
```yaml
model:
  class_path: objdet.models.yolo.YOLOv8
  init_args:
    num_classes: 80
    model_size: s  # n, s, m, l, x
    pretrained: true
    conf_thres: 0.25
    iou_thres: 0.45
```

### YOLOv11

The latest iteration of YOLO.

**Configuration:**
```yaml
model:
  class_path: objdet.models.yolo.YOLOv11
  init_args:
    num_classes: 80
    model_size: s
    pretrained: true
```

## Ensemble Methods

`objdet` provides built-in support for model ensembles to improve performance by combining predictions from multiple models.

### Weighted Box Fusion (WBF)

Often superior to NMS for ensembles.

```yaml
model:
  class_path: objdet.models.ensemble.WBFEnsemble
  init_args:
    models: [...]  # List of model configs or checkpoints
    iou_threshold: 0.55
    score_threshold: 0.1
    weights: null  # Optional weights for each model
```

### Non-Maximum Suppression (NMS)

Standard NMS ensemble.

```yaml
model:
  class_path: objdet.models.ensemble.NMSEnsemble
  init_args:
    models: [...]
    iou_threshold: 0.5
```
