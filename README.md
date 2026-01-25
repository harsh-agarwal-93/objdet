# ObjDet - Production-Grade Object Detection Training Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI/CD](https://github.com/harsh-agarwal-93/objdet/actions/workflows/ci.yml/badge.svg)](https://github.com/harsh-agarwal-93/objdet/actions/workflows/ci.yml)

A production-grade MLOps framework for training, optimizing, and deploying object detection models using PyTorch Lightning.

## 🚀 Features

- **Multiple Model Architectures**: Faster R-CNN, RetinaNet, YOLOv8, YOLOv11
- **Lightning Integration**: Full PyTorch Lightning ecosystem support with LightningCLI
- **Flexible Data Pipeline**: Support for COCO, Pascal VOC, YOLO, and custom formats
- **Optimized Datasets**: LitData integration for streamlined data loading
- **Model Ensembling**: Weighted Box Fusion, NMS, and learned ensemble strategies
- **Experiment Tracking**: MLflow and TensorBoard integration
- **Model Optimization**: TensorRT and ONNX export with SafeTensors storage
- **REST API Serving**: LitServe with dynamic batching and A/B testing
- **Large Image Inference**: SAHI integration for sliced inference
- **Job Pipelines**: Celery + RabbitMQ for distributed job submission
- **Hyperparameter Tuning**: Optuna integration for automated optimization

## 📋 Requirements

- Python 3.12+
- CUDA 12.x (for GPU training)
- [uv](https://github.com/astral-sh/uv) package manager

## 🔧 Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/harsh-agarwal-93/objdet.git
cd objdet

# Install with uv
uv sync

# Install with optional dependencies
uv sync --all-extras
```

### For Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Configure git commit template
git config commit.template .gitmessage
```

## 🏃 Quick Start

### Training a Model

```bash
# Train Faster R-CNN on COCO dataset
objdet fit --config configs/experiment/faster_rcnn_coco.yaml

# Train with LitData optimized dataset (faster data loading)
objdet fit --config configs/experiment/faster_rcnn_litdata.yaml

# Train with custom overrides
objdet fit \
    --config configs/experiment/yolov8_coco.yaml \
    --trainer.max_epochs 100 \
    --data.init_args.batch_size 16
```

### Running Inference

```bash
# Run inference on images
objdet predict \
    --config configs/experiment/yolov8_custom.yaml \
    --ckpt_path checkpoints/best.ckpt \
    --data.predict_path /path/to/images
```

### Starting the Inference Server

```bash
# Start LitServe API
objdet serve --config configs/serving/default.yaml
```

### Submitting a Training Job

```python
from objdet.pipelines.sdk import ObjDetClient

client = ObjDetClient(broker_url="amqp://rabbitmq:5672")

job = client.submit_training(
    config="configs/experiment/yolov8_coco.yaml",
    resources={"gpu": "A100", "gpu_count": 2},
)
print(f"Job submitted: {job.id}")
```

### Preprocessing Datasets

Convert datasets to LitData's optimized streaming format for faster training:

```bash
# Convert COCO dataset to LitData format
objdet preprocess \
    --input /path/to/coco \
    --output /path/to/coco_litdata \
    --format coco \
    --num_workers 8
```

Supported formats: `coco`, `voc`, `yolo`

You can also use the Python API:

```python
from objdet.data.preprocessing import convert_to_litdata

convert_to_litdata(
    input_dir="/data/coco",
    output_dir="/data/coco_litdata",
    format_name="coco",
    num_workers=8,
)
```

## 📁 Project Structure

```text
objdet/
├── configs/              # YAML configuration files
│   ├── model/            # Model-only configs
│   └── experiment/       # Full configs (model + data + trainer)
│       ├── *_coco.yaml   # COCO dataset experiments
│       └── *_litdata.yaml # LitData optimized experiments
├── src/objdet/           # Source code
│   ├── core/             # Core utilities (logging, exceptions)
│   ├── models/           # Model implementations
│   ├── data/             # Data loading and transforms
│   ├── training/         # Callbacks, metrics, losses
│   ├── optimization/     # Model export and optimization
│   ├── inference/        # Prediction utilities
│   ├── serving/          # REST API with LitServe
│   └── pipelines/        # Job queue with Celery
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## ⚠️ Class Index Handling

> **Important**: This framework makes class indexing **explicit** to avoid confusion between model formats.

- **TorchVision models** (Faster R-CNN, RetinaNet): Background class at index 0
- **YOLO models**: No background class, classes start at index 0

You must specify `class_index_mode` in your data configuration:

```yaml
data:
  class_index_mode: torchvision  # or "yolo"
```

## 📊 Logging & Experiment Tracking

Training metrics are logged to both TensorBoard and MLflow:

```bash
# View TensorBoard logs
tensorboard --logdir lightning_logs

# MLflow UI
mlflow ui --backend-store-uri mlruns
```

## 🐳 Docker

```bash
# Build training container
docker build -f Dockerfile.train -t objdet:train .

# Build serving container
docker build -f Dockerfile.serve -t objdet:serve .

# Run with docker-compose (includes RabbitMQ, MLflow)
docker-compose up -d
```

## 📖 Documentation

Full documentation is available at [https://objdet.readthedocs.io](https://objdet.readthedocs.io)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [LitServe](https://github.com/Lightning-AI/litserve)
- [LitData](https://github.com/Lightning-AI/litdata)
