from importlib import resources

version_file = resources.files(__package__).joinpath("VERSION.txt")
__version__ = version_file.read_text(encoding="utf-8").strip()

__all__ = []
