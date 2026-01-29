# Data Formats

ObjDet supports multiple data formats for object detection tasks, with optimized streaming via LitData.

## Supported Formats

| Format | DataModule | Description |
|--------|------------|-------------|
| COCO | `COCODataModule` | JSON annotations with image paths |
| Pascal VOC | `VOCDataModule` | XML annotations per image |
| YOLO | `YOLODataModule` | Text annotations per image |
| LitData | `LitDataDataModule` | Optimized streaming format |

## LitData Streaming Format

LitData provides optimized streaming for large-scale datasets with:

- **Native Streaming**: Uses `StreamingDataset` and `StreamingDataLoader` for efficient data loading
- **Cloud Integration**: Stream directly from S3, GCS, or Azure Blob Storage
- **Automatic Prefetching**: Optimized chunk-based prefetching
- **Distributed Training**: Built-in support for multi-GPU and multi-node training

### Usage

```python
from objdet.data.formats.litdata import (
    LitDataDataModule,
    DetectionStreamingDataset,
    create_streaming_dataloader,
)

# Using the DataModule (recommended)
datamodule = LitDataDataModule(
    data_dir="/data/coco_litdata",
    batch_size=16,
    num_workers=4,
)
datamodule.setup("fit")
train_loader = datamodule.train_dataloader()

# Using the dataset directly
dataset = DetectionStreamingDataset(
    input_dir="/data/coco_litdata/train",
    shuffle=True,
)

# Create dataloader with detection collation
loader = create_streaming_dataloader(
    dataset=dataset,
    batch_size=16,
    num_workers=4,
)
```

### Configuration

```yaml
data:
  class_path: objdet.data.formats.litdata.LitDataDataModule
  init_args:
    data_dir: /path/to/litdata
    train_subdir: train
    val_subdir: val
    batch_size: 16
    num_workers: 4
```

### Converting Datasets to LitData

Convert existing datasets to the optimized format:

```bash
# CLI
objdet preprocess \
    --input /path/to/coco \
    --output /path/to/coco_litdata \
    --format coco
```

```python
# Python API
from objdet.data.preprocessing import convert_to_litdata

convert_to_litdata(
    input_dir="/data/coco",
    output_dir="/data/coco_litdata",
    format_name="coco",
    num_workers=8,
)
```

## COCO Format

Standard COCO JSON format with bounding boxes.

### Expected Structure

```text
coco_dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    └── val/
```

### Usage

```python
from objdet.data.formats.coco import COCODataModule

datamodule = COCODataModule(
    data_dir="/data/coco",
    train_ann_file="annotations/instances_train.json",
    val_ann_file="annotations/instances_val.json",
)
```

## Pascal VOC Format

XML annotations with per-image files.

### Expected Structure

```text
voc_dataset/
├── Annotations/     # XML files
├── ImageSets/Main/  # train.txt, val.txt
└── JPEGImages/      # Image files
```

### Usage

```python
from objdet.data.formats.voc import VOCDataModule

datamodule = VOCDataModule(
    data_dir="/data/voc",
)
```

## YOLO Format

Text annotations with one file per image.

### Expected Structure

```text
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Label Format

Each line: `class_id center_x center_y width height` (normalized 0-1)

### Usage

```python
from objdet.data.formats.yolo import YOLODataModule

datamodule = YOLODataModule(
    data_dir="/data/yolo",
)
```

## Class Index Modes

Different model architectures expect different class indexing:

| Mode | Background | Class Range | Models |
|------|------------|-------------|--------|
| `torchvision` | Index 0 | 1 to N | Faster R-CNN, RetinaNet |
| `yolo` | None | 0 to N-1 | YOLOv8, YOLOv11 |

Specify in your config:

```yaml
data:
  class_index_mode: torchvision  # or "yolo"
```

## Custom Transforms

Apply augmentations using Albumentations:

```python
import albumentations as A

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(800, 1333),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

datamodule = COCODataModule(
    data_dir="/data/coco",
    train_transforms=train_transforms,
)
```
