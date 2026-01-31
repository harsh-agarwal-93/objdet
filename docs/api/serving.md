# Serving API

API reference for model serving utilities.

## Server

### run_server

```{eval-rst}
.. autofunction:: objdet.serving.server.run_server
```

```python
from objdet.serving import run_server

run_server(
    checkpoint_path="model.ckpt",
    host="0.0.0.0",
    port=8000,
    max_batch_size=8,
    accelerator="cuda",
    devices=1,
)
```

**CLI Usage:**

```bash
objdet serve --checkpoint model.ckpt --host 0.0.0.0 --port 8000
```

---

## API Classes

### DetectionAPI

LitServe API implementation for object detection inference.

```{eval-rst}
.. autoclass:: objdet.serving.api.DetectionAPI
   :members:
   :undoc-members:
   :show-inheritance:
```

**Request Format:**

The API accepts requests with one of the following fields:

- `image`: Base64-encoded image data
- `url`: URL to fetch image from
- `tensor`: Raw tensor data

**Response Format:**

```text
{
  "detections": [
    {
      "box": [100, 50, 300, 400],
      "label": 1,
      "score": 0.95,
      "class_name": "person"
    }
  ],
  "num_detections": 5
}
```

---

### ABTestingAPI

A/B testing wrapper for comparing multiple model versions.

```{eval-rst}
.. autoclass:: objdet.serving.api.ABTestingAPI
   :members:
   :undoc-members:
   :show-inheritance:
```

```python
from objdet.serving.api import ABTestingAPI

# Configure models with traffic weights
api = ABTestingAPI(
    models={
        "v1": ("model_v1.ckpt", 0.7),  # 70% traffic
        "v2": ("model_v2.ckpt", 0.3),  # 30% traffic
    },
    device="cuda",
)
```

**Response includes model version:**

```text
{
  "model_version": "v2",
  "detections": [{...}],
  "num_detections": 5
}
```
