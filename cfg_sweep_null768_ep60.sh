#!/bin/bash
# CFG sweep (FID5k) for the 6x768 [2] null_s_clean model, s-clean-cfg-mode=null.
# Sweeps cfg_scale on a fixed checkpoint/config; uses sample_pix_ddt_project_free_layer.py.
# Run from RAE/ root:  bash cfg_sweep_null768_ep60.sh

set -e

# NOTE: experiment dir name is the stale 1152 label, but the checkpoint inside is the
# actual 6x768 [2] null_s_clean model (67M). Leave the name as-is to match what trained.
SOURCE_EXPERIMENT=PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean
CKPT_ROOT=/home/shihaoyang/data/ckpts/PixDDTv2/${SOURCE_EXPERIMENT}
CKPT=${CKPT_ROOT}/checkpoints/ep-0000060.pt
CONFIG=configs/stage2/sampling/ImageNet256/PixDDT_free_layer_2_wo_lora_w_y_embed_adaln_zero_6_768_DiTDHS_null_s_clean.yaml
SAMPLE_BASE=/home/shihaoyang/data/samples/null768_ep60/cfg_sweep
REF=/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz
NPROC=8
NUM_FID_SAMPLES=5000
BATCH_SIZE=64
PRECISION=fp32

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results_cfg.txt"

echo "" >> "$RESULTS_FILE"
echo "--- cfg sweep (s_clean mode=null), pix_ts from config (=1.0) ---" >> "$RESULTS_FILE"
echo "ckpt: $CKPT" >> "$RESULTS_FILE"
echo "config: $CONFIG" >> "$RESULTS_FILE"
printf "%-8s  %-8s  %-8s  %-8s  %s\n" "mode" "scale" "t_min" "t_max" "FID5k" >> "$RESULTS_FILE"
echo "------------------------------------------------" >> "$RESULTS_FILE"

run_combo() {
    local mode=$1
    local scale=$2
    local t_min=$3
    local t_max=$4

    local tag
    tag=$(printf "sclean-%s-s%.2f-tmin%.2f-tmax%.2f" "$mode" "$scale" "$t_min" "$t_max")
    local npz_path="$SAMPLE_BASE/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  mode=$mode  scale=$scale  t_min=$t_min  t_max=$t_max"
    echo "  tag=$tag"
    echo "============================================================"

    if [ -f "$npz_path" ]; then
        echo "  [SKIP] npz already exists: $npz_path"
    else
        SAVE_FOLDER="$tag" \
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
            --cfg-scale "$scale" \
            --t-min "$t_min" \
            --t-max "$t_max" \
            --s-clean-cfg-mode "$mode"
    fi

    echo "  Computing FID..."
    fid_out=$(cd /home/shihaoyang/RAE/src && python -m eval.fid \
        --arr1 "$npz_path" \
        --ref "$REF" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "$fid_out" | tail -1)
    echo "  -> FID: $fid_val"
    printf "%-8s  %-8s  %-8s  %-8s  %s\n" "$mode" "$scale" "$t_min" "$t_max" "$fid_val" >> "$RESULTS_FILE"
}

# scale=1.0 → w/o CFG baseline (mode irrelevant when scale==1.0); rest are CFG on.
run_combo null 1.0 0.0 1.0
run_combo null 1.5 0.0 1.0
run_combo null 2.0 0.0 1.0
run_combo null 2.5 0.0 1.0
run_combo null 3.0 0.0 1.0

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
