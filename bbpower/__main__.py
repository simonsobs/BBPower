from __future__ import annotations

import sys

from bbpipe import PipelineStage

from ._stages import STAGE_MODULES, get_stage_class


def _print_usage() -> None:
    known = "\n- ".join(sorted(STAGE_MODULES))
    sys.stderr.write(
        "\nUsage: python -m bbpower <stage_name> <stage_arguments>\n\n"
        "Available stages:\n"
        f"- {known}\n"
    )


def main() -> int:
    """Parse the CLI arguments and run the requested pipeline stage.

    Returns
    -------
    int
        Exit code: 0 on success, 1 for usage errors, 2 for unknown stages.
    """
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_usage()
        return 1

    stage_name = sys.argv[1]
    try:
        stage_cls = get_stage_class(stage_name)
    except (ImportError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    return stage_cls.main()


if __name__ == "__main__":
    raise SystemExit(main())
