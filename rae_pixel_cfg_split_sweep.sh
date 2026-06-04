#!/bin/bash
# RAE / Pixel CFG split sweep (FID5k).
# Hypothesis: keep RAE CFG low to preserve DINO-latent realism, then apply
# stronger CFG in Pixel RGB space.
#
# Usage:
#   CKPT=/path/to/current_sota.pt bash rae_pixel_cfg_split_sweep.sh
#
# Optional overrides:
#   CONFIG=/path/to/sampling.yaml SAMPLE_BASE=/path/to/samples REF=/path/to/ref.npz \
#   REPO_ROOT=/home/shihaoyang/RAE NPROC=4 NUM_FID_SAMPLES=5000 \
#   PER_PROC_BATCH_SIZE=64 PRECISION=fp32 \
#   bash rae_pixel_cfg_split_sweep.sh

set -e

: "${CKPT:?Set CKPT=/path/to/current_sota.pt before running this sweep.}"

REPO_ROOT=${REPO_ROOT:-/home/shihaoyang/RAE}
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml}
SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/rae_pixel_cfg_split}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
NPROC=${NPROC:-4}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-5000}
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-64}
PRECISION=${PRECISION:-fp32}
BASE_SHARED_CFG=${BASE_SHARED_CFG:-1.40}

cd "$REPO_ROOT"
mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results.txt"

echo ""                                                               >> "$RESULTS_FILE"
echo "--- RAE / Pixel CFG split sweep ---"                            >> "$RESULTS_FILE"
printf "%-12s  %-10s  %-10s  %-8s  %-8s  %s\n" "round" "rae_cfg" "pix_cfg" "t_min" "t_max" "FID${NUM_FID_SAMPLES}" >> "$RESULTS_FILE"
echo "--------------------------------------------------------------------" >> "$RESULTS_FILE"

run_combo() {
    local round_name=$1
    local rae_cfg=$2
    local pix_cfg=$3
    local t_min=$4
    local t_max=$5
    local tag
    tag=$(printf "%s-rae%.2f-pix%.2f-tmin%.1f-tmax%.1f" "$round_name" "$rae_cfg" "$pix_cfg" "$t_min" "$t_max")
    local npz_path="$SAMPLE_BASE/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  round=$round_name  rae_cfg=$rae_cfg  pix_cfg=$pix_cfg  t_min=$t_min  t_max=$t_max  tag=$tag"
    echo "============================================================"

    if [ -f "$npz_path" ]; then
        echo "  [SKIP] npz already exists: $npz_path"
    else
        SAVE_FOLDER="$tag" \
        torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
            src/sample_pix_ddt_project_free_layer.py \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --sample-dir "$SAMPLE_BASE" \
            --num-fid-samples "$NUM_FID_SAMPLES" \
            --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
            --precision "$PRECISION" \
            --label-sampling equal \
            --naive \
            --cfg-scale "$BASE_SHARED_CFG" \
            --rae-cfg-scale "$rae_cfg" \
            --pix-cfg-scale "$pix_cfg" \
            --t-min "$t_min" \
            --t-max "$t_max"
    fi

    echo "  Computing FID..."
    fid_out=$(cd "$REPO_ROOT/src" && python -m eval.fid \
        --arr1 "$npz_path" \
        --ref "$REF" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "$fid_out" | tail -1)
    echo "  -> FID: $fid_val"
    printf "%-12s  %-10s  %-10s  %-8s  %-8s  %s\n" "$round_name" "$rae_cfg" "$pix_cfg" "$t_min" "$t_max" "$fid_val" >> "$RESULTS_FILE"
}

# Round 1: isolate CFG split at the current best shared interval.
# Keep one shared t interval for both phases so the only variable is CFG strength.
run_combo cfg_base "$BASE_SHARED_CFG" "$BASE_SHARED_CFG" 0.0 1.0

run_combo cfg_split 1.0 1.5 0.0 1.0
run_combo cfg_split 1.0 2.0 0.0 1.0
run_combo cfg_split 1.0 2.5 0.0 1.0
run_combo cfg_split 1.2 2.0 0.0 1.0
run_combo cfg_split 1.5 2.0 0.0 1.0
run_combo cfg_split 1.5 2.5 0.0 1.0

# Round 2: small t-interval probe for likely winners.
# cfg_sweep.sh suggests t_max=1.0 should stay fixed; only probe t_min.
run_combo t_probe 1.0 2.0 0.1 1.0
run_combo t_probe 1.0 2.0 0.2 1.0
run_combo t_probe 1.0 2.0 0.3 1.0
run_combo t_probe 1.0 2.5 0.1 1.0
run_combo t_probe 1.0 2.5 0.2 1.0
run_combo t_probe 1.0 2.5 0.3 1.0

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
