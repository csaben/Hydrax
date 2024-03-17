from importlib import resources

from .core import get_trained_models, train
from .jax_filters import init_models, parallel_init
from .loss import loss
from .unbatching import get_original_model, unbatch_model

version_file = resources.files(__package__).joinpath("VERSION.txt")
__version__ = version_file.read_text(encoding="utf-8").strip()

__all__ = [
    "train",
    "get_trained_models",
    "get_original_model",
    "unbatch_model",
    "loss",
    "init_models",
    "parallel_init",
]
