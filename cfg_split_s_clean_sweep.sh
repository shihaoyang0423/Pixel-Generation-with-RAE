#!/bin/bash
# Split s_clean CFG ablation (FID5k).
# Compares shared vs split s_clean conditioning on the same checkpoint/config.

set -e

SOURCE_EXPERIMENT=PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDH-XL
CKPT_ROOT=/home/shihaoyang/data/ckpts/PixDDTv2/${SOURCE_EXPERIMENT}
CKPT=${CKPT_ROOT}/checkpoints/ep-0000050.pt
CONFIG=${CKPT_ROOT}/config.yaml
SAMPLE_BASE=/home/shihaoyang/data/samples/${SOURCE_EXPERIMENT}/cfg_split_s_clean_fid5k_ep0000050
REF=/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz
NPROC=4
NUM_FID_SAMPLES=5000
BATCH_SIZE=64
PRECISION=fp32

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results_split_s_clean.txt"

echo "" >> "$RESULTS_FILE"
echo "--- split_s_clean CFG ablation ---" >> "$RESULTS_FILE"
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

# First pass: keep t_min/t_max fixed near the known CFG sweep region.
run_combo shared 2.50 0.30 1.00

run_combo split 1.50 0.30 1.00
run_combo split 1.75 0.30 1.00
run_combo split 2.00 0.30 1.00
run_combo split 2.25 0.30 1.00
run_combo split 2.50 0.30 1.00

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
