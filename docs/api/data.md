# Data API

API reference for data modules and datasets.

## LitData Streaming

### LitDataDataModule

::: objdet.data.formats.litdata.LitDataDataModule
    options:
      show_root_heading: true
      members:
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader

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

## Standard Formats

### COCODataModule

::: objdet.data.formats.coco.COCODataModule
    options:
      show_root_heading: true

### COCODataset

::: objdet.data.formats.coco.COCODataset
    options:
      show_root_heading: true

### VOCDataModule

::: objdet.data.formats.voc.VOCDataModule
    options:
      show_root_heading: true

### YOLODataModule

::: objdet.data.formats.yolo.YOLODataModule
    options:
      show_root_heading: true

## Base Classes

### BaseDataModule

::: objdet.data.base.BaseDataModule
    options:
      show_root_heading: true
      members:
        - __init__
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader

### detection_collate_fn

::: objdet.data.base.detection_collate_fn
    options:
      show_root_heading: true

## Data Preprocessing

### convert_to_litdata

::: objdet.data.preprocessing.litdata_converter.convert_to_litdata
    options:
      show_root_heading: true

### LitDataConverter

::: objdet.data.preprocessing.litdata_converter.LitDataConverter
    options:
      show_root_heading: true
