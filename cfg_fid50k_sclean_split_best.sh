#!/bin/bash
# FID50k verification for the best FID5k CFG point found so far:
# cfg_scale=1.50, t_min=0.0, t_max=1.0, split s_clean, pix_time_shift=3.0.

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDH-XL}
CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints/ep-0000050.pt}
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/PixDDT_free_layer_1_3_5_7_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_pix_ts3.yaml}
SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXPERIMENT_NAME}/fid50k_cfg1.50_t0.0_1.0_sclean_split_pix_ts3}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
PRECISION=${PRECISION:-fp32}
CFG_SCALE=${CFG_SCALE:-1.50}
T_MIN=${T_MIN:-0.0}
T_MAX=${T_MAX:-1.0}
S_CLEAN_CFG_MODE=${S_CLEAN_CFG_MODE:-split}

if [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found: $CKPT" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 1
fi
if [ ! -f "$REF" ]; then
    echo "[error] reference stats not found: $REF" >&2
    exit 1
fi

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid50k_result.txt"
TAG="cfg-s${CFG_SCALE}-tmin${T_MIN}-tmax${T_MAX}-sclean-${S_CLEAN_CFG_MODE}-n${NUM_FID_SAMPLES}"
NPZ_PATH="$SAMPLE_BASE/${TAG}.npz"

echo "--- FID50k CFG verification ---" | tee -a "$RESULTS_FILE"
echo "experiment: $EXPERIMENT_NAME" | tee -a "$RESULTS_FILE"
echo "ckpt: $CKPT" | tee -a "$RESULTS_FILE"
echo "config: $CONFIG" | tee -a "$RESULTS_FILE"
echo "sample_base: $SAMPLE_BASE" | tee -a "$RESULTS_FILE"
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
        --s-clean-cfg-mode "$S_CLEAN_CFG_MODE"
fi

echo "Computing FID..." | tee -a "$RESULTS_FILE"
fid_out=$(cd /home/shihaoyang/RAE/src && python -m eval.fid \
    --arr1 "$NPZ_PATH" \
    --ref "$REF" \
    --ref-mu-key mu \
    --ref-sigma-key sigma 2>&1)
fid_val=$(echo "$fid_out" | tail -1)
echo "$fid_out" | tee -a "$RESULTS_FILE"
echo "FID50k: $fid_val" | tee -a "$RESULTS_FILE"
