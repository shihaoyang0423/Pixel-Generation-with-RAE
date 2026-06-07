#!/bin/bash
# 4-mode s_clean CFG comparison (FID5k) for the 6x768 [2] null_s_clean model.
# Runs --s-clean-cfg-mode in {shared, split, null, traj} at fixed cfg scales,
# so one `bash` gives all four experiments side by side.
# Uses sample_pix_ddt_project_free_layer.py.  Run from RAE/ root:
#   bash cfg_sweep_null768_ep60.sh

set -e

# NOTE: experiment dir name is the stale 1152 label, but the checkpoint inside is the
# actual 6x768 [2] null_s_clean model (67M). Leave the name as-is to match what trained.
SOURCE_EXPERIMENT=PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean
CKPT_ROOT=/home/shihaoyang/data/ckpts/PixDDTv2/${SOURCE_EXPERIMENT}
CKPT=${CKPT_ROOT}/checkpoints/ep-0000060.pt
CONFIG=configs/stage2/sampling/ImageNet256/PixDDT_free_layer_2_wo_lora_w_y_embed_adaln_zero_6_768_DiTDHS_null_s_clean.yaml
SAMPLE_BASE=/home/shihaoyang/data/samples/null768_ep60/mode_compare
REF=/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz
NPROC=8
NUM_FID_SAMPLES=5000
BATCH_SIZE=64
PRECISION=fp32

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results_modes.txt"

echo "" >> "$RESULTS_FILE"
echo "--- s_clean mode compare + RAE-CFG on/off, pix_ts from config (=1.0) ---" >> "$RESULTS_FILE"
echo "ckpt: $CKPT" >> "$RESULTS_FILE"
echo "config: $CONFIG" >> "$RESULTS_FILE"
printf "%-8s  %-8s  %-8s  %-8s  %-8s  %s\n" "mode" "pixcfg" "t_min" "t_max" "raecfg" "FID5k" >> "$RESULTS_FILE"
echo "------------------------------------------------------------" >> "$RESULTS_FILE"

# run_combo MODE PIX_CFG_SCALE T_MIN T_MAX [RAE_CFG_SCALE]
# 5th arg optional: empty → RAE-CFG coupled to pix scale (current behaviour);
#                   1.0   → RAE-CFG OFF (Phase 1 = single natural conditional latent).
run_combo() {
    local mode=$1
    local scale=$2
    local t_min=$3
    local t_max=$4
    local rae_cfg=${5:-}

    local rae_args=()
    local rae_tag=""
    local rae_col="coupled"
    if [ -n "$rae_cfg" ]; then
        rae_args=(--rae-cfg-scale "$rae_cfg")
        rae_tag="-$(printf 'rae%.2f' "$rae_cfg")"
        rae_col="$rae_cfg"
    fi

    local tag
    tag=$(printf "sclean-%s-s%.2f-tmin%.2f-tmax%.2f%s" "$mode" "$scale" "$t_min" "$t_max" "$rae_tag")
    local npz_path="$SAMPLE_BASE/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  mode=$mode  pixcfg=$scale  t_min=$t_min  t_max=$t_max  raecfg=$rae_col"
    echo "  tag=$tag"
    echo "============================================================"

    if [ -f "$npz_path" ]; then
        echo "  [SKIP] npz already exists (no resample): $npz_path"
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
            --s-clean-cfg-mode "$mode" \
            "${rae_args[@]}"
    fi

    echo "  Computing FID..."
    fid_out=$(cd /home/shihaoyang/RAE/src && python -m eval.fid \
        --arr1 "$npz_path" \
        --ref "$REF" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "$fid_out" | tail -1)
    echo "  -> FID: $fid_val"
    printf "%-8s  %-8s  %-8s  %-8s  %-8s  %s\n" "$mode" "$scale" "$t_min" "$t_max" "$rae_col" "$fid_val" >> "$RESULTS_FILE"
}

# (1) 4-mode compare, RAE-CFG coupled = ON (already computed last run → npz cached, no resample).
for mode in shared split null traj; do
    run_combo "$mode" 2.0 0.0 1.0
    run_combo "$mode" 2.5 0.0 1.0
done

# (2) NEW: RAE-CFG OFF (rae_cfg=1.0), Pixel-CFG on. Phase 1 = single natural conditional
#     latent; tests whether the un-sharpened latent gives a more in-distribution s_clean.
for mode in null split; do
    run_combo "$mode" 1.5 0.0 1.0 1.0
    run_combo "$mode" 2.0 0.0 1.0 1.0
    run_combo "$mode" 2.5 0.0 1.0 1.0
    run_combo "$mode" 3.0 0.0 1.0 1.0
done

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
