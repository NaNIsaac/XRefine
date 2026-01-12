"""
XRefine Models
==============
PINN 모델 및 공통 컴포넌트 Export
"""
from .common import (
    BasePINNModel,
    FourierFeatureMapping,
    PositionalEncoding,
    PatchEmbed1D,
)
from .fourier_model import XRR_Fourier_PINN, DilatedCNNEncoder
from .transformer import XRR_Transformer_PINN, TransformerVAEEncoder
from .vit import XRR_ViT_PINN, ViTEncoder

# Model Registry (Factory Pattern)
MODEL_REGISTRY = {
    'cnn': XRR_Fourier_PINN,
    'transformer': XRR_Transformer_PINN,
    'vit': XRR_ViT_PINN,
}


def get_model(config):
    """
    Config에서 model_type을 읽어 해당 모델 인스턴스 반환.
    
    Usage:
        model = get_model(config)
    """
    model_type = config.model.model_type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](config)


__all__ = [
    # Base
    "BasePINNModel",
    "FourierFeatureMapping", 
    "PositionalEncoding",
    "PatchEmbed1D",
    # Models
    "XRR_Fourier_PINN",
    "XRR_Transformer_PINN", 
    "XRR_ViT_PINN",
    # Encoders
    "DilatedCNNEncoder",
    "TransformerVAEEncoder",
    "ViTEncoder",
    # Factory
    "get_model",
    "MODEL_REGISTRY",
]
