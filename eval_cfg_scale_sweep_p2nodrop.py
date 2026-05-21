#!/usr/bin/env python3
"""
eval_cfg_scale_sweep_p2nodrop.py — CFG scale sweep for the Phase-2-no-dropout experiment.

Mirrors eval_cfg_scale_sweep.py but targets the p2nodrop checkpoint and config.
Scale values are read from cfg_sweep.scales in the base YAML; the YAML is never
modified — each run overrides guidance.scale via --cfg-scale.

Experiment context
------------------
The p2nodrop model (PixelDDTv2_project_free_layer_p2nodrop.PixelDDT) was trained
with Phase-2 class dropout disabled.  This aligns the training distribution with
CFG inference:
  Training:   Phase2 = y_cond (always)   Phase3 = y_cond (90%) or y_null (10%)
  Inference:  Phase2 = y_cond (always)   Phase3 = y_cond (cond) or y_null (uncond)

Compare results against eval_cfg_scale_sweep.py (base model) to measure whether
the aligned dropout improves CFG quality.

Usage (from RAE/ root, after training is done):
    python eval_cfg_scale_sweep_p2nodrop.py

Outputs per scale value:
    PNG dir : {SAMPLE_ROOT}/cfg_s{scale:.2f}/
    NPZ file: {SAMPLE_ROOT}/cfg_s{scale:.2f}.npz
    log     : {SAMPLE_ROOT}/cfg_s{scale:.2f}.log
Summary   : {SAMPLE_ROOT}/results.txt

Resumable: if a .npz already exists for a scale, sampling is skipped and FID is
recomputed directly from the existing file.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import yaml  # PyYAML (available in the rae conda env)

# =============================================================================
# USER SETTINGS — fill in after the p2nodrop model finishes training
# =============================================================================

CKPT = (
    "/path/to/p2nodrop/checkpoints/ep-XXXXXXX.pt"   # <-- fill in
)

BASE_CONFIG = (
    "configs/stage2/sampling/ImageNet256"
    "/PixDDT_small_v2_lora_project_free_layer_p2nodrop_cfg_sweep.yaml"
)

SAMPLE_ROOT = os.path.expanduser(
    "~/data/samples/cfg_scale_sweep_p2nodrop"   # <-- adjust as needed
)

REF_NPZ = os.path.expanduser("~/data/data/VIRTUAL_imagenet256_labeled.npz")

NPROC_PER_NODE = 4    # number of GPUs
BATCH_SIZE = 64       # --per-proc-batch-size
NUM_FID_SAMPLES = 5000
PRECISION = "fp32"

PYTHON = "/home/shihaoyang/miniconda3/envs/rae/bin/python"

# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)


def _run(cmd: list[str], log_path: str, extra_env: dict | None = None) -> int:
    """Run cmd, streaming output to both stdout and log_path. Returns exit code."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
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
    """Run eval.fid and return the FID result line (e.g. '[RESULT] FID: 43.50')."""
    env_full = os.environ.copy()
    env_full["PYTHONPATH"] = str(_SCRIPT_DIR / "src")
    cmd = [
        PYTHON, "-m", "eval.fid",
        "--arr1", npz_path,
        "--ref", REF_NPZ,
        "--ref-mu-key", "mu",
        "--ref-sigma-key", "sigma",
    ]
    result_lines = []
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
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
    for line in reversed(result_lines):
        if "[RESULT]" in line or "FID:" in line:
            return line
    return f"ERROR (exit {proc.returncode}): " + " ".join(result_lines[-3:])


def sweep() -> None:
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    scales: list = base_cfg.get("cfg_sweep", {}).get("scales", [])
    if not scales:
        raise ValueError(
            f"cfg_sweep.scales not found or empty in {BASE_CONFIG}."
        )

    sample_root = Path(SAMPLE_ROOT)
    sample_root.mkdir(parents=True, exist_ok=True)
    results_file = sample_root / "results.txt"

    num_k = NUM_FID_SAMPLES // 1000
    print(f"\n{'='*60}")
    print(f"CFG Scale FID{num_k}k Sweep — p2nodrop experiment")
    print(f"Checkpoint : {CKPT}")
    print(f"Config     : {BASE_CONFIG}")
    print(f"Scales     : {scales}")
    print(f"Output root: {sample_root}")
    print(f"{'='*60}\n")

    for scale in scales:
        scale = float(scale)
        tag = f"cfg_s{scale:.2f}"
        npz_path = str(sample_root / f"{tag}.npz")
        log_path = str(sample_root / f"{tag}.log")

        print(f"\n{'─'*60}")
        print(f"[sweep] scale={scale:.2f}  tag={tag}")
        print(f"{'─'*60}")

        if os.path.isfile(npz_path):
            print(f"[sweep] NPZ already exists, skipping sampling: {npz_path}")
        else:
            sample_cmd = [
                "torchrun",
                "--standalone", "--nnodes=1",
                f"--nproc_per_node={NPROC_PER_NODE}",
                "src/sample_pix_ddt_project_free_layer.py",
                "--config", BASE_CONFIG,
                "--ckpt", CKPT,
                "--sample-dir", str(sample_root),
                "--num-fid-samples", str(NUM_FID_SAMPLES),
                "--per-proc-batch-size", str(BATCH_SIZE),
                "--precision", PRECISION,
                "--label-sampling", "equal",
                "--naive",
                "--cfg-scale", str(scale),
            ]
            print(f"[sweep] Running sampling...")
            rc = _run(
                sample_cmd,
                log_path=log_path,
                extra_env={"SAVE_FOLDER": tag},
            )
            if rc != 0:
                msg = (
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"{tag}  scale={scale:.2f}  SAMPLING FAILED (rc={rc})\n"
                )
                print(msg.rstrip())
                with open(results_file, "a") as rf:
                    rf.write(msg)
                continue

        print(f"[sweep] Computing FID from: {npz_path}")
        fid_result = _compute_fid(npz_path, log_path)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = f"[{timestamp}] {tag}  scale={scale:.2f}  {fid_result}\n"
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
