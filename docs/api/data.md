# Data API

API reference for data modules and datasets.

## LitData Streaming

### LitDataDataModule

```{eval-rst}
.. autoclass:: objdet.data.formats.litdata.LitDataDataModule
   :members: setup, train_dataloader, val_dataloader, test_dataloader
   :undoc-members:
   :show-inheritance:
```

### DetectionStreamingDataset

Factory function that creates a `StreamingDataset` with detection transforms applied.

```python
from objdet.data.formats.litdata import DetectionStreamingDataset

dataset = DetectionStreamingDataset(
    input_dir="/data/coco_litdata/train",
    shuffle=True,
    drop_last=False,
    transforms=None,  # Optional additional transforms
)
```

**Parameters:**

- `input_dir` (str | Path): Directory containing LitData optimized chunks
- `shuffle` (bool): Whether to shuffle the data (default: False)
- `drop_last` (bool): Whether to drop the last incomplete batch (default: False)
- `transforms` (Callable | None): Optional transform to apply after detection format conversion

**Returns:** A `litdata.StreamingDataset` configured for object detection.

### create_streaming_dataloader

```python
from objdet.data.formats.litdata import create_streaming_dataloader

loader = create_streaming_dataloader(
    dataset=streaming_dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)
```

**Parameters:**

- `dataset`: A StreamingDataset instance
- `batch_size` (int): Batch size for the dataloader
- `num_workers` (int): Number of worker processes (default: 4)
- `pin_memory` (bool): Whether to pin memory for faster GPU transfer (default: True)
- `drop_last` (bool): Whether to drop the last incomplete batch (default: False)

**Returns:** A `litdata.StreamingDataLoader` with detection-specific collation.

---

## Standard Formats

### COCODataModule

```{eval-rst}
.. autoclass:: objdet.data.formats.coco.COCODataModule
   :members:
   :undoc-members:
   :show-inheritance:
```

### COCODataset

```{eval-rst}
.. autoclass:: objdet.data.formats.coco.COCODataset
   :members:
   :undoc-members:
   :show-inheritance:
```

### VOCDataModule

```{eval-rst}
.. autoclass:: objdet.data.formats.voc.VOCDataModule
   :members:
   :undoc-members:
   :show-inheritance:
```

### YOLODataModule

```{eval-rst}
.. autoclass:: objdet.data.formats.yolo.YOLODataModule
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Base Classes

### BaseDataModule

```{eval-rst}
.. autoclass:: objdet.data.base.BaseDataModule
   :members: __init__, setup, train_dataloader, val_dataloader, test_dataloader
   :undoc-members:
   :show-inheritance:
```

### detection_collate_fn

```{eval-rst}
.. autofunction:: objdet.data.base.detection_collate_fn
```

---

## Data Preprocessing

### convert_to_litdata

```{eval-rst}
.. autofunction:: objdet.data.preprocessing.litdata_converter.convert_to_litdata
```

### LitDataConverter

```{eval-rst}
.. autoclass:: objdet.data.preprocessing.litdata_converter.LitDataConverter
   :members:
   :undoc-members:
   :show-inheritance:
```
