#!/usr/bin/env python3
"""
eval_pix_ts_sweep.py — Sweep over pixel_time_shift values and compute FID5k for each.

pixel_time_shift is derived from YAML misc.time_dist_shift_dim / time_dist_shift_base:
    time_dist_shift_pix = sqrt(time_dist_shift_dim / time_dist_shift_base)
This script dynamically patches the YAML for each target shift value, runs torchrun
sampling, then computes FID5k, and writes a summary to results.txt.

Usage (from RAE/ root):
    python eval_pix_ts_sweep.py

Outputs per pixel_time_shift value:
    PNG dir : {SAMPLE_ROOT}/ag-s{scale:.2f}-pix_ts{value:.2f}/
    NPZ file: {SAMPLE_ROOT}/ag-s{scale:.2f}-pix_ts{value:.2f}.npz
    log     : {SAMPLE_ROOT}/ag-s{scale:.2f}-pix_ts{value:.2f}.log
Summary   : {EXPERIMENT_ROOT}/results.txt  (parent of SAMPLE_ROOT)
"""

import copy
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml  # PyYAML (already available in the rae env)

# =============================================================================
# USER SETTINGS — edit this section to control the sweep
# =============================================================================

# List of pixel_time_shift values to test for the 6x768 AG run.
# pix_ts = sqrt(misc.time_dist_shift_dim / misc.time_dist_shift_base).
PIXEL_TIME_SHIFTS = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]  # <-- modify here

CKPT = (
    "/home/shihaoyang/data/ckpts/PixDDTv2/"
    "PixDDT_small_v2_project_free_layer_2_false_logit_normal_0.8_0.8_pix_ts_1_wo_lora_w_y_embed_adaln_zero"
    "/checkpoints/ep-0000050.pt"
)

# Base YAML (relative to RAE/ root) — misc.time_dist_shift_dim will be patched per run
BASE_CONFIG = (
    "configs/stage2/sampling/ImageNet256"
    "/PixDDT_6x768_project_free_layer_2_false_pix_ts1_AG.yaml"
)

# Fixed AutoGuidance settings for this pix_ts sweep.
GUIDANCE_METHOD = "autoguidance"
AG_PHASE = "condition"
AG_CKPT = (
    "/home/shihaoyang/data/pretrained/DiTs/Dinov2/wReg_base/ImageNet256/"
    "DiTDH-S_ep14/stage2_model.pt"
)
AG_SCALE = 3.0
T_MIN = 0.0
T_MAX = 1.0

# Experiment root: results.txt is written here
EXPERIMENT_ROOT = os.path.expanduser(
    "~/data/samples/"
    "PixDDT_small_v2_project_free_layer_2_false_logit_normal_0.8_0.8_pix_ts_1_wo_lora_w_y_embed_adaln_zero"
    "_autoguidance_ditdh_condition"
)

# Sweep outputs (PNG dirs, NPZ files, logs) go under this subdirectory
SAMPLE_ROOT = os.path.join(EXPERIMENT_ROOT, f"pix_ts_sweep_ag_scale_{AG_SCALE:.2f}")

# Reference statistics NPZ for FID computation
REF_NPZ = os.path.expanduser("~/data/data/VIRTUAL_imagenet256_labeled.npz")

NPROC_PER_NODE = 8    # number of GPUs / torchrun processes
BATCH_SIZE = 64       # --per-proc-batch-size
NUM_FID_SAMPLES = 5000
PRECISION = "fp32"

# Python executable with torch / numpy etc. (rae conda env)
PYTHON = "/home/shihaoyang/miniconda3/envs/rae/bin/python"

# =============================================================================

# Script must run from RAE/ root so that relative paths (src/, configs/) work.
_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)


def _ts_to_dim(ts: float, base: int = 4096) -> int:
    """Reverse of sqrt(dim/base): dim = ts² * base."""
    return int(round(ts * ts * base))


def _build_tmp_config(ts: float, base_cfg: dict) -> str:
    """
    Patch a copy of base_cfg (YAML snapshot) for misc.time_dist_shift_dim,
    write to a named temp file, and return its path.
    The caller is responsible for deleting it.
    """
    cfg = copy.deepcopy(base_cfg)

    base = cfg.get("misc", {}).get("time_dist_shift_base", 4096)
    new_dim = _ts_to_dim(ts, base)
    cfg.setdefault("misc", {})["time_dist_shift_dim"] = new_dim

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"pix_ts{ts:.2f}_cfg_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _run(cmd: list[str], log_path: str, extra_env: dict | None = None) -> int:
    """
    Run cmd, streaming output to both stdout and log_path.
    Returns the process exit code.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        proc.wait()
    return proc.returncode


def _compute_fid(npz_path: str, log_path: str) -> str:
    """
    Run eval.fid and return the result string (e.g. '[RESULT] FID: 43.50').
    """
    env = {"PYTHONPATH": str(_SCRIPT_DIR / "src")}
    cmd = [
        PYTHON, "-m", "eval.fid",
        "--arr1", npz_path,
        "--ref", REF_NPZ,
        "--ref-mu-key", "mu",
        "--ref-sigma-key", "sigma",
    ]
    env_full = os.environ.copy()
    env_full.update(env)

    result_lines = []
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_full,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            result_lines.append(line.rstrip())
        proc.wait()

    # Extract the FID result line (starts with [RESULT])
    for line in reversed(result_lines):
        if "[RESULT]" in line or "FID:" in line:
            return line
    return f"ERROR (exit {proc.returncode}): " + " ".join(result_lines[-3:])


def sweep():
    experiment_root = Path(EXPERIMENT_ROOT)
    sample_root = Path(SAMPLE_ROOT)
    sample_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    results_file = experiment_root / "results.txt"

    required_paths = {
        "CKPT": CKPT,
        "BASE_CONFIG": BASE_CONFIG,
        "AG_CKPT": AG_CKPT,
        "REF_NPZ": REF_NPZ,
        "PYTHON": PYTHON,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n  " + "\n  ".join(missing))

    print(f"\n{'='*60}")
    print(f"Pixel TS FID5k Sweep")
    print(f"Checkpoint : {CKPT}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"AG ckpt    : {AG_CKPT}")
    print(f"AG scale   : {AG_SCALE}")
    print(f"Values     : {PIXEL_TIME_SHIFTS}")
    print(f"Sample root: {sample_root}")
    print(f"Results    : {results_file}")
    print(f"{'='*60}\n")

    with open(BASE_CONFIG) as f:
        base_cfg_snapshot = yaml.safe_load(f)

    for ts in PIXEL_TIME_SHIFTS:
        ts_tag = f"ag-s{AG_SCALE:.2f}-pix_ts{ts:.2f}"
        dim = _ts_to_dim(ts)
        npz_path = str(sample_root / f"{ts_tag}.npz")
        log_path = str(sample_root / f"{ts_tag}.log")

        print(f"\n{'─'*60}")
        print(f"[sweep] ts={ts:.2f}  dim={dim}  tag={ts_tag}")
        print(f"{'─'*60}")

        if ts <= 0.0:
            print(f"[sweep] SKIP: ts={ts:.2f} <= 0 is invalid")
            continue

        # --- Sampling (skip if npz already exists) ---
        if os.path.isfile(npz_path):
            print(f"[sweep] NPZ already exists, skipping sampling: {npz_path}")
        else:
            tmp_cfg = _build_tmp_config(ts, base_cfg_snapshot)
            try:
                sample_cmd = [
                    "torchrun",
                    "--standalone", "--nnodes=1",
                    f"--nproc_per_node={NPROC_PER_NODE}",
                    "src/sample_pix_ddt_project_free_layer.py",
                    "--config", tmp_cfg,
                    "--ckpt", CKPT,
                    "--sample-dir", str(sample_root),
                    "--num-fid-samples", str(NUM_FID_SAMPLES),
                    "--per-proc-batch-size", str(BATCH_SIZE),
                    "--precision", PRECISION,
                    "--label-sampling", "equal",
                    "--naive",
                    "--guidance-method", GUIDANCE_METHOD,
                    "--ag-phase", AG_PHASE,
                    "--ag-ckpt", AG_CKPT,
                    "--cfg-scale", str(AG_SCALE),
                    "--t-min", str(T_MIN),
                    "--t-max", str(T_MAX),
                ]
                print(f"[sweep] Running sampling...")
                rc = _run(
                    sample_cmd,
                    log_path=log_path,
                    extra_env={"SAVE_FOLDER": ts_tag},
                )
                if rc != 0:
                    msg = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {ts_tag}  dim={dim}  SAMPLING FAILED (rc={rc})\n"
                    print(msg.rstrip())
                    with open(results_file, "a") as rf:
                        rf.write(msg)
                    continue
            finally:
                os.unlink(tmp_cfg)

        # --- FID computation ---
        print(f"[sweep] Computing FID from: {npz_path}")
        fid_result = _compute_fid(npz_path, log_path)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = f"[{timestamp}] {ts_tag}  dim={dim}  {fid_result}\n"
        print(f"[sweep] {record.rstrip()}")
        with open(results_file, "a") as rf:
            rf.write(record)

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results written to: {results_file}")
    print(f"{'='*60}\n")
    # Print final summary
    if results_file.exists():
        print(results_file.read_text())


if __name__ == "__main__":
    sweep()
