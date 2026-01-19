"""ObjDet - Production-grade object detection training framework.

A modular, extensible framework for training, optimizing, and deploying
object detection models using PyTorch Lightning.

Example:
    >>> from objdet.models import FasterRCNN
    >>> from objdet.data import COCODataModule
    >>> from lightning import Trainer
    >>>
    >>> model = FasterRCNN(num_classes=80)
    >>> datamodule = COCODataModule(data_dir="/path/to/coco")
    >>> trainer = Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
"""

from objdet.version import __version__

__all__ = ["__version__"]
