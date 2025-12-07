"""Top-level package for llm_lab."""

from typing import Any

from ._version import __version__


# Lazy imports to avoid circular dependencies during installation
def __getattr__(name: str) -> Any:
    if name == "cli_app":
        from .cli import app
        return app
    elif name == "ExperimentPaths":
        from .config import ExperimentPaths
        return ExperimentPaths
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "ExperimentPaths",
    "cli_app",
]
