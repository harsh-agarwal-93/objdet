# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

## Install from PyPI

For usage as a CLI tool:

```bash
uv tool install objdet
```

For usage as a library in a project:

```bash
uv add objdet
```

## Install from Source

```bash
git clone https://github.com/example/objdet.git
cd objdet
uv sync --all-extras
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
uv add "objdet[tensorrt]"
```

### Optimization

For Lightning Thunder compilation:

```bash
uv add "objdet[thunder]"
```

### All Features

```bash
uv add "objdet[all]"
```
