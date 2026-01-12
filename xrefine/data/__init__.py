"""
XRefine Data Module
==================
Synthetic data generation and augmentation utilities.
"""
from xrefine.data.generator import generate_synthetic_batch, XRRStreamingDataset
from xrefine.data.augmentations import XRRAugmentor

__all__ = ["generate_synthetic_batch", "XRRStreamingDataset", "XRRAugmentor"]
