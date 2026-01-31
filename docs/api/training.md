# Training API

API reference for training utilities.

## Callbacks

Custom Lightning callbacks for object detection training.

### ConfusionMatrixCallback

```{eval-rst}
.. autoclass:: objdet.training.callbacks.confusion_matrix.ConfusionMatrixCallback
   :members:
   :undoc-members:
   :show-inheritance:
```

Generates and saves confusion matrix visualizations during validation.

```python
from objdet.training import ConfusionMatrixCallback
from lightning import Trainer

trainer = Trainer(
    callbacks=[
        ConfusionMatrixCallback(
            num_classes=80,
            class_names=class_names,
            save_dir="./confusion_matrices",
        ),
    ],
)
```

---

### DetectionVisualizationCallback

```{eval-rst}
.. autoclass:: objdet.training.callbacks.visualization.DetectionVisualizationCallback
   :members:
   :undoc-members:
   :show-inheritance:
```

Visualizes detection predictions on sample images during training.

```python
from objdet.training import DetectionVisualizationCallback

callback = DetectionVisualizationCallback(
    num_samples=8,
    score_threshold=0.5,
    class_names=class_names,
)
```

---

### GradientMonitorCallback

```{eval-rst}
.. autoclass:: objdet.training.callbacks.gradient_monitor.GradientMonitorCallback
   :members:
   :undoc-members:
   :show-inheritance:
```

Monitors and logs gradient statistics during training.

```python
from objdet.training import GradientMonitorCallback

callback = GradientMonitorCallback(
    log_every_n_steps=100,
)
```

---

### LearningRateMonitorCallback

```{eval-rst}
.. autoclass:: objdet.training.callbacks.lr_monitor.LearningRateMonitorCallback
   :members:
   :undoc-members:
   :show-inheritance:
```

Enhanced learning rate monitoring with additional logging.

```python
from objdet.training import LearningRateMonitorCallback

callback = LearningRateMonitorCallback()
```

---

## Metrics

Custom metrics for object detection evaluation.

### ClasswiseAP

```{eval-rst}
.. autoclass:: objdet.training.metrics.classwise_ap.ClasswiseAP
   :members:
   :undoc-members:
   :show-inheritance:
```

Computes per-class Average Precision.

```python
from objdet.training.metrics import ClasswiseAP

metric = ClasswiseAP(
    num_classes=80,
    class_names=class_names,
    iou_threshold=0.5,
)

# Update with predictions and targets
metric.update(predictions, targets)

# Compute results
results = metric.compute()
# {"class_0_AP": 0.85, "class_1_AP": 0.72, ...}
```

---

### ConfusionMatrix

```{eval-rst}
.. autoclass:: objdet.training.metrics.confusion_matrix.ConfusionMatrix
   :members:
   :undoc-members:
   :show-inheritance:
```

Computes detection confusion matrix.

```python
from objdet.training.metrics import ConfusionMatrix

metric = ConfusionMatrix(
    num_classes=80,
    iou_threshold=0.5,
    conf_threshold=0.25,
)

metric.update(predictions, targets)
matrix = metric.compute()  # Returns (num_classes+1, num_classes+1) tensor
```

```{note}
The matrix includes an extra row/column for background (false positives/negatives).
```
