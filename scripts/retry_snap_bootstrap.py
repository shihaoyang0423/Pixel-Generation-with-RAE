"""
Register snapshotted train/sample/model modules in sys.modules, then run the
snapshotted entry script. Used by retry_train_stage3_project_free_layer.sh so
only the frozen copies are used while all other imports still come from the repo.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import sys

SNAP_SRC = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.environ["RAE_SRC"]
RAE_ROOT = os.environ["RAE_ROOT"]

for p in (RAE_SRC, RAE_ROOT):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _install_snap(qualified_name: str, filename: str) -> None:
    path = os.path.join(SNAP_SRC, filename)
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load snap module {qualified_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)


def _install_common_snaps() -> None:
    _install_snap(
        "stage2.models.PixelDDTv2_project_free_layer",
        "PixelDDTv2_project_free_layer.py",
    )
    _install_snap("train_stage3_projcet_free_layer", "train_stage3_projcet_free_layer.py")


def main() -> None:
    mode = os.environ.get("RETRY_SNAP_MODE", "train")
    _install_common_snaps()
    if mode == "train":
        entry = os.path.join(SNAP_SRC, "train_stage3_projcet_free_layer.py")
    elif mode == "sample":
        _install_snap("sample_pix_ddt_project_free_layer", "sample_pix_ddt_project_free_layer.py")
        entry = os.path.join(SNAP_SRC, "sample_pix_ddt_project_free_layer.py")
    else:
        raise ValueError(f"Unknown RETRY_SNAP_MODE: {mode!r}")
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
