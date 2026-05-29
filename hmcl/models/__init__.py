
from typing import Any

from torch.nn import Module
from hmcl.models.conv import NoiseConditionedConvClassifier
from hmcl.models.mlp import NoiseConditionedMLP, BaseMLP, ResidualMLP

def build_model(model_name: str, input_dim: int, target_dim: int, **kwargs: Any) -> Module:
    model_name = model_name.lower()
    if model_name == "base_mlp":
        return BaseMLP(input_dim=input_dim, target_dim=target_dim, **kwargs)
    if model_name == "noise_conditioned_mlp":
        return NoiseConditionedMLP(input_dim=input_dim, target_dim=target_dim, **kwargs)
    if model_name == "residual_mlp":
        return ResidualMLP(input_dim=input_dim, target_dim=target_dim, **kwargs)
    raise ValueError("model_name must be 'base_mlp' or 'noise_conditioned_mlp'.")

__all__ = [
    "NoiseConditionedConvClassifier", 
    "NoiseConditionedMLP", 
    "BaseMLP", 
    "build_dataset",
]
