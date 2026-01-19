# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

## Install from PyPI

```bash
pip install objdet
```

## Install from Source

```bash
git clone https://github.com/example/objdet.git
cd objdet
pip install -e ".[dev]"
```

## Install with uv (Recommended)

```bash
uv pip install objdet
```

Or for development:

```bash
uv pip install -e ".[dev,docs]"
```

## Verify Installation

```python
import objdet
print(objdet.__version__)
```

## Optional Dependencies

### GPU Acceleration

For TensorRT optimization:

```bash
pip install "objdet[tensorrt]"
```

### Serving

For LitServe deployment:

```bash
pip install "objdet[serve]"
```

### All Features

```bash
pip install "objdet[all]"
```
