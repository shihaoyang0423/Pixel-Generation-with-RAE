#!/bin/bash
# CPU all-metrics evaluation for the best 16x1536 layer-6 null_s_clean single point.
# Defaults: cfg_scale=1.2, pix_ts=3.5, s_clean=null, t=[0.0,1.0], ep50.
#
# By default this script evaluates an existing 50k sample NPZ on CPU.
# Set RUN_SAMPLE=1 to first run cfg_fid50k.sh with the same point, then evaluate metrics.
#
# Run from RAE/ root:
#   bash cfg_all_metrics.sh
#   RUN_SAMPLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 bash cfg_all_metrics.sh
#   CFG_SCALE=1.5 PIX_TS=3.0 bash cfg_all_metrics.sh

set -euo pipefail

EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean}
EP=${EP:-0000050}
CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints/ep-${EP}.pt}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
PYTHON=${PYTHON:-/home/shihaoyang/miniconda3/envs/rae/bin/python}

CFG_SCALE=${CFG_SCALE:-1.2}
T_MIN=${T_MIN:-0.0}
T_MAX=${T_MAX:-1.0}
S_CLEAN_CFG_MODE=${S_CLEAN_CFG_MODE:-null}
PIX_TS=${PIX_TS:-3.5}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
RAE_CFG_SCALE=${RAE_CFG_SCALE:-}
RUN_SAMPLE=${RUN_SAMPLE:-0}

PIX_TS_TAG="-pixts${PIX_TS}"
RAE_TAG=""
if [ -n "$RAE_CFG_SCALE" ]; then
    RAE_TAG="-rae${RAE_CFG_SCALE}"
fi

SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXPERIMENT_NAME}/fid50k_ep${EP}_cfg${CFG_SCALE}_sclean-${S_CLEAN_CFG_MODE}${PIX_TS_TAG}${RAE_TAG}}
TAG="cfg-s${CFG_SCALE}-tmin${T_MIN}-tmax${T_MAX}-sclean-${S_CLEAN_CFG_MODE}${PIX_TS_TAG}${RAE_TAG}-n${NUM_FID_SAMPLES}"
NPZ_PATH=${NPZ_PATH:-$SAMPLE_BASE/${TAG}.npz}
RESULTS_FILE=${RESULTS_FILE:-$SAMPLE_BASE/all_metrics_result.txt}

if [ ! -f "$CKPT" ]; then echo "[error] checkpoint not found: $CKPT" >&2; exit 1; fi
if [ ! -f "$REF" ]; then echo "[error] reference batch not found: $REF" >&2; exit 1; fi

if [ "$RUN_SAMPLE" = "1" ]; then
    echo "[sample] generating/reusing 50k NPZ via cfg_fid50k.sh"
    CFG_SCALE="$CFG_SCALE" \
    T_MIN="$T_MIN" \
    T_MAX="$T_MAX" \
    S_CLEAN_CFG_MODE="$S_CLEAN_CFG_MODE" \
    PIX_TS="$PIX_TS" \
    NUM_FID_SAMPLES="$NUM_FID_SAMPLES" \
    RAE_CFG_SCALE="$RAE_CFG_SCALE" \
    SAMPLE_BASE="$SAMPLE_BASE" \
    CKPT="$CKPT" \
    REF="$REF" \
    PYTHON="$PYTHON" \
        bash cfg_fid50k.sh
fi

if [ ! -f "$NPZ_PATH" ]; then
    echo "[error] sample NPZ not found: $NPZ_PATH" >&2
    echo "        Run cfg_fid50k.sh first, or use RUN_SAMPLE=1 when GPUs are available." >&2
    exit 1
fi

mkdir -p "$SAMPLE_BASE"

echo "--- All metrics single point ---" | tee -a "$RESULTS_FILE"
echo "experiment: $EXPERIMENT_NAME  (ep${EP})" | tee -a "$RESULTS_FILE"
echo "ckpt: $CKPT" | tee -a "$RESULTS_FILE"
echo "ref: $REF" | tee -a "$RESULTS_FILE"
echo "sample: $NPZ_PATH" | tee -a "$RESULTS_FILE"
echo "point: cfg=$CFG_SCALE  t=[$T_MIN,$T_MAX]  mode=$S_CLEAN_CFG_MODE  pix_ts=$PIX_TS  rae_cfg=${RAE_CFG_SCALE:-coupled}  n=$NUM_FID_SAMPLES" | tee -a "$RESULTS_FILE"
echo "device: CPU (CUDA_VISIBLE_DEVICES forced empty for metrics)" | tee -a "$RESULTS_FILE"

echo "Computing IS / FID / Precision / Recall..." | tee -a "$RESULTS_FILE"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2} \
    "$PYTHON" src/eval/all_metrics_torch_fidelity.py \
        --ref "$REF" \
        --sample "$NPZ_PATH" \
        --batch-size "${METRIC_BATCH_SIZE:-64}" \
        --device cpu 2>&1 | tee -a "$RESULTS_FILE"
