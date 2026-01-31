# Inference API

API reference for inference utilities.

## Predictor

High-level inference predictor for running detection on images.

```{eval-rst}
.. autoclass:: objdet.inference.predictor.Predictor
   :members:
   :undoc-members:
   :show-inheritance:
```

### Usage Example

```python
from objdet.inference import Predictor

# Create from checkpoint
predictor = Predictor.from_checkpoint(
    checkpoint_path="model.ckpt",
    device="cuda",
    confidence_threshold=0.25,
)

# Single image inference
result = predictor.predict("image.jpg")
print(f"Found {len(result['boxes'])} objects")

# Batch inference
results = predictor.predict_batch(["img1.jpg", "img2.jpg"], batch_size=8)

# Directory inference
results_dict = predictor.predict_directory("./images", extensions=(".jpg", ".png"))
```

---

## SlicedInference (SAHI)

Slicing Aided Hyper Inference for detecting small objects in large images.

```{eval-rst}
.. autoclass:: objdet.inference.sahi_wrapper.SlicedInference
   :members:
   :undoc-members:
   :show-inheritance:
```

### Usage Example

```python
from objdet.inference import Predictor, SlicedInference

# Create base predictor
predictor = Predictor.from_checkpoint("model.ckpt")

# Wrap with sliced inference
sahi = SlicedInference(
    predictor=predictor,
    slice_height=640,
    slice_width=640,
    overlap_ratio=0.2,
    merge_method="nms",  # or "wbf"
    include_full_image=True,
)

# Run sliced inference on large image
result = sahi.predict("large_satellite_image.jpg")
```

**Parameters:**

- `predictor`: Base Predictor instance for running inference on slices
- `slice_height` (int): Height of each slice (default: 640)
- `slice_width` (int): Width of each slice (default: 640)
- `overlap_ratio` (float): Overlap between adjacent slices (default: 0.2)
- `merge_method` (str): Method for merging overlapping predictions - "nms" or "wbf" (default: "nms")
- `nms_threshold` (float): IoU threshold for NMS merging (default: 0.5)
- `include_full_image` (bool): Also run inference on full image (default: True)
