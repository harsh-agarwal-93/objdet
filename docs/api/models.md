# Models API

API reference for object detection models.

## Model Registry

The global registry for all detection models. Use this to build models by name.

```{eval-rst}
.. autodata:: objdet.models.registry.MODEL_REGISTRY
   :annotation:
```

### Registered Models

| Name | Aliases | Class |
|------|---------|-------|
| `faster_rcnn` | `fasterrcnn`, `frcnn` | `FasterRCNN` |
| `retinanet` | - | `RetinaNet` |
| `yolov8` | `yolo8` | `YOLOv8` |
| `yolov11` | `yolo11` | `YOLOv11` |

```python
from objdet.models import MODEL_REGISTRY

# Build model from registry
model = MODEL_REGISTRY.build("faster_rcnn", num_classes=80)
```

---

## Base Class

### BaseLightningDetector

```{eval-rst}
.. autoclass:: objdet.models.base.BaseLightningDetector
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## TorchVision Models

### FasterRCNN

Two-stage detector with Region Proposal Network.

```{eval-rst}
.. autoclass:: objdet.models.torchvision.faster_rcnn.FasterRCNN
   :members:
   :undoc-members:
   :show-inheritance:
```

```python
from objdet.models.torchvision import FasterRCNN
from lightning import Trainer

model = FasterRCNN(
    num_classes=80,
    backbone="resnet50_fpn_v2",
    pretrained_backbone=True,
    trainable_backbone_layers=3,
)

trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

---

### RetinaNet

One-stage detector with focal loss.

```{eval-rst}
.. autoclass:: objdet.models.torchvision.retinanet.RetinaNet
   :members:
   :undoc-members:
   :show-inheritance:
```

```python
from objdet.models.torchvision import RetinaNet

model = RetinaNet(
    num_classes=80,
    backbone="resnet50_fpn_v2",
    pretrained_backbone=True,
    score_thresh=0.05,
    nms_thresh=0.5,
)
```

---

## YOLO Models

### YOLOv8

```{eval-rst}
.. autoclass:: objdet.models.yolo.yolov8.YOLOv8
   :members:
   :undoc-members:
   :show-inheritance:
```

**Model Sizes:**

| Size | Variant | Parameters |
|------|---------|------------|
| n (nano) | `yolov8n.pt` | ~3.2M |
| s (small) | `yolov8s.pt` | ~11.2M |
| m (medium) | `yolov8m.pt` | ~25.9M |
| l (large) | `yolov8l.pt` | ~43.7M |
| x (extra-large) | `yolov8x.pt` | ~68.2M |

```python
from objdet.models.yolo import YOLOv8

model = YOLOv8(
    num_classes=80,
    model_size="m",
    pretrained=True,
    conf_thres=0.25,
    iou_thres=0.45,
)
```

```{warning}
There is a known bug in the training pipeline that causes
`IndexError: too many indices for tensor of dimension 2` during loss computation.
```

---

### YOLOv11

```{eval-rst}
.. autoclass:: objdet.models.yolo.yolov11.YOLOv11
   :members:
   :undoc-members:
   :show-inheritance:
```

**Model Sizes:**

| Size | Variant |
|------|---------|
| n (nano) | `yolo11n.pt` |
| s (small) | `yolo11s.pt` |
| m (medium) | `yolo11m.pt` |
| l (large) | `yolo11l.pt` |
| x (extra-large) | `yolo11x.pt` |

```python
from objdet.models.yolo import YOLOv11

model = YOLOv11(
    num_classes=80,
    model_size="l",
    pretrained=True,
)
```

```{warning}
YOLOv11 has the same known training bug as YOLOv8.
```
