from __future__ import annotations

from importlib import import_module
from typing import Any

from bbpipe import PipelineStage  # noqa

from ._stages import STAGE_MODULES

__all__ = ["PipelineStage", *STAGE_MODULES]


def __getattr__(name: str) -> Any:
    module_name = STAGE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    return getattr(module, name)
