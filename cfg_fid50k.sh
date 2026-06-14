#!/bin/bash
# Single-point FID50k for the 16x1536 layer-6 null_s_clean model.
# Defaults to the best FID5k point from cfg_sweep:
#   cfg_scale=1.2, pix_ts=3.5, s_clean=null, t=[0.0,1.0], ep50.
#
# Run from RAE/ root:
#   bash cfg_fid50k.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash cfg_fid50k.sh
#   CFG_SCALE=1.5 PIX_TS=3.0 bash cfg_fid50k.sh

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean}
EP=${EP:-0000050}
CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints/ep-${EP}.pt}
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
PYTHON=${PYTHON:-/home/shihaoyang/miniconda3/envs/rae/bin/python}

if [ -z "${NPROC:-}" ]; then
    NPROC=$("$PYTHON" -c "import torch; print(max(torch.cuda.device_count(), 1))")
fi
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
PRECISION=${PRECISION:-fp32}

CFG_SCALE=${CFG_SCALE:-1.2}
T_MIN=${T_MIN:-0.0}
T_MAX=${T_MAX:-1.0}
S_CLEAN_CFG_MODE=${S_CLEAN_CFG_MODE:-null}
PIX_TS=${PIX_TS:-3.5}

# RAE_CFG_SCALE: empty -> couple RAE-CFG to CFG_SCALE; set 1.0 to turn RAE-CFG off.
RAE_CFG_SCALE=${RAE_CFG_SCALE:-}

if [ ! -f "$CKPT" ]; then echo "[error] checkpoint not found: $CKPT" >&2; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "[error] config not found: $CONFIG" >&2; exit 1; fi
if [ ! -f "$REF" ]; then echo "[error] reference stats not found: $REF" >&2; exit 1; fi

# pix_ts maps to misc.time_dist_shift_dim = round(PIX_TS^2 * 4096), pixel stream only.
dim=$("$PYTHON" -c "print(round($PIX_TS**2*4096))")
PATCHED_CONFIG=/tmp/fid50k_1536_ep${EP}_pixts${PIX_TS}.yaml
"$PYTHON" - "$CONFIG" "$PATCHED_CONFIG" "$dim" <<'PY'
import sys, yaml
src, dst, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("misc", {})["time_dist_shift_dim"] = dim
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY
CONFIG="$PATCHED_CONFIG"
PIX_TS_TAG="-pixts${PIX_TS}"
echo "[pix_ts] PIX_TS=$PIX_TS (dim=$dim), patched config: $PATCHED_CONFIG"

RAE_ARGS=()
RAE_TAG=""
if [ -n "$RAE_CFG_SCALE" ]; then
    RAE_ARGS=(--rae-cfg-scale "$RAE_CFG_SCALE")
    RAE_TAG="-rae${RAE_CFG_SCALE}"
fi

SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXPERIMENT_NAME}/fid50k_ep${EP}_cfg${CFG_SCALE}_sclean-${S_CLEAN_CFG_MODE}${PIX_TS_TAG}${RAE_TAG}}
mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid50k_result.txt"
TAG="cfg-s${CFG_SCALE}-tmin${T_MIN}-tmax${T_MAX}-sclean-${S_CLEAN_CFG_MODE}${PIX_TS_TAG}${RAE_TAG}-n${NUM_FID_SAMPLES}"
NPZ_PATH="$SAMPLE_BASE/${TAG}.npz"

echo "--- FID50k single point ---" | tee -a "$RESULTS_FILE"
echo "experiment: $EXPERIMENT_NAME  (ep${EP})" | tee -a "$RESULTS_FILE"
echo "ckpt: $CKPT" | tee -a "$RESULTS_FILE"
echo "config: $CONFIG" | tee -a "$RESULTS_FILE"
echo "point: cfg=$CFG_SCALE  t=[$T_MIN,$T_MAX]  mode=$S_CLEAN_CFG_MODE  pix_ts=$PIX_TS  rae_cfg=${RAE_CFG_SCALE:-coupled}  n=$NUM_FID_SAMPLES" | tee -a "$RESULTS_FILE"
echo "nproc: $NPROC" | tee -a "$RESULTS_FILE"
echo "tag: $TAG" | tee -a "$RESULTS_FILE"

if [ -f "$NPZ_PATH" ]; then
    echo "[SKIP] npz already exists: $NPZ_PATH" | tee -a "$RESULTS_FILE"
else
    SAVE_FOLDER="$TAG" \
    torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
        src/sample_pix_ddt_project_free_layer.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --sample-dir "$SAMPLE_BASE" \
        --num-fid-samples "$NUM_FID_SAMPLES" \
        --per-proc-batch-size "$BATCH_SIZE" \
        --precision "$PRECISION" \
        --label-sampling equal \
        --naive \
        --cfg-scale "$CFG_SCALE" \
        --t-min "$T_MIN" \
        --t-max "$T_MAX" \
        --s-clean-cfg-mode "$S_CLEAN_CFG_MODE" \
        "${RAE_ARGS[@]}"
fi

echo "Computing FID..." | tee -a "$RESULTS_FILE"
fid_out=$(cd /home/shihaoyang/RAE/src && "$PYTHON" -m eval.fid \
    --arr1 "$NPZ_PATH" \
    --ref "$REF" \
    --ref-mu-key mu \
    --ref-sigma-key sigma 2>&1)
echo "$fid_out" | tee -a "$RESULTS_FILE"
fid_val=$(echo "$fid_out" | tail -1)
echo "FID50k: $fid_val" | tee -a "$RESULTS_FILE"
