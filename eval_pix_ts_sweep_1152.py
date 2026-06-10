#!/usr/bin/env python3
"""
eval_pix_ts_sweep_1152.py — Sweep pixel_time_shift and compute FID5k for each,
for the 12x1152 [1,3,5,7] null_s_clean model (any checkpoint; pick epoch via the EP env var).

Adapted from Xiuyuan/CFG_split_s_clean's eval_pix_ts_sweep.py. pixel_time_shift is
derived from misc.time_dist_shift_dim / time_dist_shift_base:
    pix_ts = sqrt(time_dist_shift_dim / time_dist_shift_base)
This script patches the YAML's misc.time_dist_shift_dim for each target shift, runs
torchrun sampling, then computes FID5k and writes a summary to results.txt.

Usage (from RAE/ root):
    python eval_pix_ts_sweep_1152.py             # default epoch
    EP=0000080 python eval_pix_ts_sweep_1152.py  # any checkpoint

Outputs per pixel_time_shift value:
    PNG dir : {SAMPLE_ROOT}/pix_ts{value:.2f}/
    NPZ file: {SAMPLE_ROOT}/pix_ts{value:.2f}.npz
    log     : {SAMPLE_ROOT}/pix_ts{value:.2f}.log
Summary   : {EXPERIMENT_ROOT}/results.txt
"""

import copy
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

# List of pixel_time_shift values to test. The 1152 sampling config bakes pix_ts=3.0,
# so probe around it. (GPU-tight first look — trim this list to go faster.)
PIXEL_TIME_SHIFTS = [1.0, 2.0, 3.0, 4.0, 6.0]  # <-- modify here

SOURCE_EXPERIMENT = (
    "PixDDT_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_"
    "12_1152_DiTDHXL_null_s_clean"
)
EP = os.environ.get("EP", "0000050")   # override per checkpoint: EP=0000080 python ...
CKPT = (
    f"/home/shihaoyang/data/ckpts/PixDDTv2/{SOURCE_EXPERIMENT}"
    f"/checkpoints/ep-{EP}.pt"
)

# Base YAML (relative to RAE/ root) — misc.time_dist_shift_dim is patched per run.
BASE_CONFIG = (
    "configs/stage2/sampling/ImageNet256"
    "/PixDDT_free_layer_1_3_5_7_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean_pix_ts3.yaml"
)

# Experiment root: results.txt is written here.
EXPERIMENT_ROOT = os.path.expanduser(
    f"~/data/samples/{SOURCE_EXPERIMENT}/pix_ts_sweep_ep{EP}"
)
# Sweep outputs (PNG dirs, NPZ files, logs) go under this subdirectory.
SAMPLE_ROOT = os.path.join(EXPERIMENT_ROOT, "compare_different_pix_ts")

# Reference statistics NPZ for FID computation.
REF_NPZ = os.path.expanduser("~/data/data/VIRTUAL_imagenet256_labeled.npz")

NPROC_PER_NODE = 8    # number of GPUs / torchrun processes
BATCH_SIZE = 64       # --per-proc-batch-size
NUM_FID_SAMPLES = 5000
PRECISION = "fp32"

# CFG / s_clean conditioning passed to the sampler.
# CFG_SCALE = None → no CFG override (use config's guidance.scale). Set a float to sweep
# pix_ts at a fixed CFG strength. S_CLEAN_CFG_MODE = "null" matches how this model trained.
CFG_SCALE = 2.0
S_CLEAN_CFG_MODE = "null"
RAE_CFG_SCALE = None   # None = couple RAE-CFG to CFG_SCALE; set e.g. 1.0 to turn RAE-CFG off

# Python executable with torch / numpy etc. (rae conda env)
PYTHON = "/home/shihaoyang/miniconda3/envs/rae/bin/python"

# =============================================================================

# Script must run from RAE/ root so relative paths (src/, configs/) work.
_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)


def _ts_to_dim(ts: float, base: int = 4096) -> int:
    """Reverse of sqrt(dim/base): dim = ts^2 * base."""
    return int(round(ts * ts * base))


def _build_tmp_config(ts: float, base_cfg: dict) -> str:
    """Patch a copy of base_cfg for misc.time_dist_shift_dim; write to a temp file."""
    cfg = copy.deepcopy(base_cfg)
    base = cfg.get("misc", {}).get("time_dist_shift_base", 4096)
    cfg.setdefault("misc", {})["time_dist_shift_dim"] = _ts_to_dim(ts, base)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"pix_ts{ts:.2f}_1152_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _run(cmd: list[str], log_path: str, extra_env: dict | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        proc.wait()
    return proc.returncode


def _compute_fid(npz_path: str, log_path: str) -> str:
    cmd = [
        PYTHON, "-m", "eval.fid",
        "--arr1", npz_path,
        "--ref", REF_NPZ,
        "--ref-mu-key", "mu",
        "--ref-sigma-key", "sigma",
    ]
    env_full = os.environ.copy()
    env_full.update({"PYTHONPATH": str(_SCRIPT_DIR / "src")})

    result_lines = []
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env_full, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            result_lines.append(line.rstrip())
        proc.wait()

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

    print(f"\n{'='*60}")
    print(f"Pixel TS FID5k Sweep — 12x1152 [1,3,5,7] null_s_clean @ ep{EP}")
    print(f"Checkpoint : {CKPT}")
    print(f"Values     : {PIXEL_TIME_SHIFTS}")
    print(f"CFG        : scale={CFG_SCALE}  s_clean_mode={S_CLEAN_CFG_MODE}  rae_cfg={RAE_CFG_SCALE}")
    print(f"Sample root: {sample_root}")
    print(f"Results    : {results_file}")
    print(f"{'='*60}\n")

    with open(BASE_CONFIG) as f:
        base_cfg_snapshot = yaml.safe_load(f)

    for ts in PIXEL_TIME_SHIFTS:
        ts_tag = f"pix_ts{ts:.2f}"
        dim = _ts_to_dim(ts)
        npz_path = str(sample_root / f"{ts_tag}.npz")
        log_path = str(sample_root / f"{ts_tag}.log")

        print(f"\n{'-'*60}")
        print(f"[sweep] ts={ts:.2f}  dim={dim}  tag={ts_tag}")
        print(f"{'-'*60}")

        if ts <= 0.0:
            print(f"[sweep] SKIP: ts={ts:.2f} <= 0 is invalid")
            continue

        if os.path.isfile(npz_path):
            print(f"[sweep] NPZ already exists, skipping sampling: {npz_path}")
        else:
            tmp_cfg = _build_tmp_config(ts, base_cfg_snapshot)
            try:
                sample_cmd = [
                    "torchrun", "--standalone", "--nnodes=1",
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
                ]
                if CFG_SCALE is not None:
                    sample_cmd += ["--cfg-scale", str(CFG_SCALE)]
                if S_CLEAN_CFG_MODE is not None:
                    sample_cmd += ["--s-clean-cfg-mode", S_CLEAN_CFG_MODE]
                if RAE_CFG_SCALE is not None:
                    sample_cmd += ["--rae-cfg-scale", str(RAE_CFG_SCALE)]
                print(f"[sweep] Running sampling...")
                rc = _run(sample_cmd, log_path=log_path, extra_env={"SAVE_FOLDER": ts_tag})
                if rc != 0:
                    msg = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {ts_tag}  dim={dim}  SAMPLING FAILED (rc={rc})\n"
                    print(msg.rstrip())
                    with open(results_file, "a") as rf:
                        rf.write(msg)
                    continue
            finally:
                os.unlink(tmp_cfg)

        print(f"[sweep] Computing FID from: {npz_path}")
        fid_result = _compute_fid(npz_path, log_path)
        record = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {ts_tag}  dim={dim}  {fid_result}\n"
        print(f"[sweep] {record.rstrip()}")
        with open(results_file, "a") as rf:
            rf.write(record)

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results written to: {results_file}")
    print(f"{'='*60}\n")
    if results_file.exists():
        print(results_file.read_text())


if __name__ == "__main__":
    sweep()
