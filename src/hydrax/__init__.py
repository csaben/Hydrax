from importlib import resources

from .core import get_trained_models, train
from .unbatching import get_original_model, unbatch_model
from .loss import loss

version_file = resources.files(__package__).joinpath("VERSION.txt")
__version__ = version_file.read_text(encoding="utf-8").strip()

__all__ = ["train", "get_trained_models", "get_original_model", "unbatch_model", "loss"]
