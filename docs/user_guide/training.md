# Training

Training in `objdet` is handled via the CLI, which acts as a wrapper around [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

## The `fit` Command

The primary command for training is `objdet fit`.

```bash
objdet fit --config configs/experiment/faster_rcnn_coco.yaml
```

## Configuration Structure

Our configurations follow a composed structure:

1.  **Model**: Defines architecture, optimizer, and scheduler.
2.  **Data**: Defines dataset, batch size, and transforms.
3.  **Trainer**: Defines PyTorch Lightning trainer flags (epochs, GPUs, callbacks).

### Experiment Configs

Experiment configs in `configs/experiment` combine these sections for reproducible runs.

**Example `faster_rcnn_coco.yaml`:**

```yaml
# @package _global_

defaults:
  - /model: faster_rcnn
  - /data: coco
  - /trainer: default

trainer:
  max_epochs: 12
  accelerator: gpu
  devices: 1

data:
  batch_size: 4
```

## Running Experiments

To run an experiment:

```bash
# Basic run
objdet fit --config configs/experiment/yolov8_coco.yaml

# Override parameters
objdet fit --config configs/experiment/yolov8_coco.yaml \
    --trainer.max_epochs 50 \
    --data.batch_size 16 \
    --model.init_args.learning_rate 0.005
```

## Multi-GPU Training

`objdet` supports distributed training out of the box.

```bash
# Train on 2 GPUs using DDP
objdet fit --config configs/experiment/faster_rcnn_coco.yaml \
    --trainer.devices 2 \
    --trainer.strategy ddp
```

## Resume Training

To resume from a checkpoint:

```bash
objdet fit --config configs/experiment/faster_rcnn_coco.yaml \
    --ckpt_path training_logs/version_0/checkpoints/last.ckpt
```
