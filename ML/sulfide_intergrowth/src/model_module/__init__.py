"""Model factory & registry. Models are config-driven: __init__(cfg) only."""
from typing import Callable, Dict, Type

import torch.nn as nn

MODEL_FACTORY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator registering a model under `name`."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_FACTORY[name] = cls
        return cls
    return decorator


def ModelFactory(name: str, cfg) -> nn.Module:
    """Instantiate a registered model from its config node.

    Raises:
        KeyError: if the model name is not registered.
    """
    if name not in MODEL_FACTORY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_FACTORY)}")
    return MODEL_FACTORY[name](cfg)


from . import classifier, segmenter  # noqa: E402,F401  (populate the registry)

__all__ = ["MODEL_FACTORY", "ModelFactory", "register_model"]
