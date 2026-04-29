from __future__ import annotations

from importlib import import_module
from typing import Any


STAGE_MODULES: dict[str, str] = {
    "BBPowerSpecter": "bbpower.power_specter",
    "BBPowerSummarizer": "bbpower.power_summarizer",
    "BBCompSep": "bbpower.compsep",
    "BBPlotter": "bbpower.plotter",
}


def get_stage_class(stage_name: str) -> Any:
    """Import and return the pipeline stage class for *stage_name*.

    Parameters
    ----------
    stage_name : str
        Registered name of the stage (e.g. ``'BBCompSep'``).

    Returns
    -------
    type
        The ``PipelineStage`` subclass.

    Raises
    ------
    KeyError
        If *stage_name* is not in ``STAGE_MODULES``.
    """
    try:
        module_name = STAGE_MODULES[stage_name]
    except KeyError as exc:
        known = ", ".join(sorted(STAGE_MODULES))
        raise KeyError(
            f"Unknown BBPower stage {stage_name!r}. Known stages: {known}"
        ) from exc

    module = import_module(module_name)
    return getattr(module, stage_name)
