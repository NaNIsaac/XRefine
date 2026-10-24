"""
XRefine Data Module
==================
Synthetic data generation and augmentation utilities.
"""

from xrefine.data.augmentations import XRRAugmentor
from xrefine.data.generator import XRRStreamingDataset, generate_synthetic_batch

__all__ = ["generate_synthetic_batch", "XRRStreamingDataset", "XRRAugmentor"]
