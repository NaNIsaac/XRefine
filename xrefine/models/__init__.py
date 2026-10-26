"""
XRefine Models Registry
=======================
Decorator-based model registration for easy extension.
"""

from .common import BasePINNModel

# Registry for PINN models
MODEL_REGISTRY: dict[str, type[BasePINNModel]] = {}


def register_model(name: str):
    """Decorator to register a model class in the registry."""

    def decorator(cls: type[BasePINNModel]):
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_model(config) -> BasePINNModel:
    """Factory function to create a model instance from config."""
    model_type = config.model.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        # Import all models to ensure registration
        # Importing a model registers it, so this order is the order of MODEL_REGISTRY.
        from . import vit, transformer, fourier_model  # noqa: F401, I001

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_type](config)


# Exporting components
from .common import (
    BasePINNModel,
    FourierFeatureMapping,
    PatchEmbed1D,
    PositionalEncoding,
)

__all__ = [
    "MODEL_REGISTRY",
    "BasePINNModel",
    "FourierFeatureMapping",
    "PatchEmbed1D",
    "PositionalEncoding",
    "get_model",
    "register_model",
]
