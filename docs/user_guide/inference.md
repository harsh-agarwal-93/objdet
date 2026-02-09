# Inference

`objdet` provides both a Python API and a CLI for running inference on trained models.

## Python API

The `Predictor` class is the main entry point for inference.

### Basic Usage

```python
from objdet.inference import Predictor

# Load from checkpoint
predictor = Predictor.from_checkpoint(
    checkpoint_path="checkpoints/model.ckpt",
    device="cuda",
    confidence_threshold=0.3
)

# Predict single image
result = predictor.predict("images/test.jpg")
print(f"Found {len(result['boxes'])} objects")

# Predict batch
results = predictor.predict_batch(["img1.jpg", "img2.jpg"])
```

### Sliced Inference (SAHI)

For large images (e.g., satellite/aerial), use `SlicedInference` to detect small objects by processing image tiles.

```python
from objdet.inference import SlicedInference

# Initialize SAHI wrapper around predictor
sahi = SlicedInference(
    predictor=predictor,
    slice_height=640,
    slice_width=640,
    overlap_ratio=0.2,
    merge_method="nms"  # or "wbf"
)

# Run inference
result = sahi.predict("large_image.tiff")
```

## Command Line Interface

You can also run inference directly from the terminal.

### `objdet predict`

Run inference on an image or directory.

```bash
objdet predict \
    --config configs/model/faster_rcnn.yaml \
    --ckpt_path checkpoints/model.ckpt \
    --data.predict_data_path data/test_images \
    --trainer.accelerator gpu
```

### `objdet serve`

Start a REST API for model serving (using LitServe).

```bash
objdet serve --config configs/serving/default.yaml
```

## Output Format

Predictions are returned as a dictionary:

- **boxes**: `(N, 4)` tensor of bounding boxes `[x1, y1, x2, y2]`.
- **labels**: `(N,)` tensor of class indices.
- **scores**: `(N,)` tensor of confidence scores.
