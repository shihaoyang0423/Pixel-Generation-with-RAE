#!/bin/bash
# Sampling sweep (FID5k) for the 12x1152 [1,3,5,7] null_s_clean model (any checkpoint; pick epoch via EP=...).
# Mirrors Xiuyuan/CFG_split_s_clean's cfg_sweep_null768_ep60.sh, adapted to the new
# from-scratch 12x1152 / DiTDH-XL checkpoint. Uses sample_pix_ddt_project_free_layer.py.
#
# Run from RAE/ root:
#   bash cfg_sweep_1152_null_s_clean.sh                # default block: no-CFG baseline +
#                                                           #   null cfg-scale sweep + 4-mode compare
#   DECOUPLE=1 bash cfg_sweep_1152_null_s_clean.sh     # RAE<->Pixel CFG decouple (rae-pixel-cfg-split)
#   INTERVAL=1 bash cfg_sweep_1152_null_s_clean.sh     # CFG-interval [t_min,t_max] sweep
#   PIX_TS=6  bash cfg_sweep_1152_null_s_clean.sh      # override pixel_time_shift (default = config's 3.0)
#   EP=0000040 bash ...                                     # evaluate a different epoch checkpoint
#
# Env knobs (with defaults): EP=0000050, NPROC=8, NUM_FID_SAMPLES=5000, BATCH_SIZE=64.

set -e

SOURCE_EXPERIMENT=${SOURCE_EXPERIMENT:-PixDDT_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean}
CKPT_ROOT=/home/shihaoyang/data/ckpts/PixDDTv2/${SOURCE_EXPERIMENT}
EP=${EP:-0000050}
CKPT=${CKPT:-${CKPT_ROOT}/checkpoints/ep-${EP}.pt}
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/PixDDT_free_layer_1_3_5_7_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean_pix_ts3.yaml}

# Optional pixel_time_shift override. Empty → use the config's baked pix_ts (=3.0).
# Patches misc.time_dist_shift_dim = round(PIX_TS^2 * 4096) via a robust YAML edit
# (the 1152 config already ships dim=36864, so a literal sed on 4096 would not match).
PIX_TS=${PIX_TS:-}
# INTERVAL=1 → run a CFG-interval sweep (restrict CFG to [t_min,t_max]) instead of the
# mode/decouple blocks. Composes with PIX_TS. (t=1 noise end, t=0 clean end.)
INTERVAL=${INTERVAL:-}
# DECOUPLE=1 → run the RAE<->Pixel CFG decouple block (--rae-cfg-scale) instead.
DECOUPLE=${DECOUPLE:-}

SAMPLE_BASE=/home/shihaoyang/data/samples/${SOURCE_EXPERIMENT}/sampling_ep${EP}
REF=/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz
PYTHON=${PYTHON:-/home/shihaoyang/miniconda3/envs/rae/bin/python}
NPROC=${NPROC:-8}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-5000}
BATCH_SIZE=${BATCH_SIZE:-64}
PRECISION=${PRECISION:-fp32}

# If PIX_TS is set, build a patched config and tag outputs so they don't collide with the
# config-default pix_ts runs.
PIX_TS_TAG=""
if [ -n "$PIX_TS" ]; then
    dim=$("$PYTHON" -c "print(round($PIX_TS**2*4096))")
    PATCHED_CONFIG=/tmp/cfg_sweep_1152_ep${EP}_pixts${PIX_TS}.yaml
    "$PYTHON" - "$CONFIG" "$PATCHED_CONFIG" "$dim" <<'PY'
import sys, yaml
src, dst, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("misc", {})["time_dist_shift_dim"] = dim   # only the pixel stream
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY
    CONFIG="$PATCHED_CONFIG"
    PIX_TS_TAG="-pixts${PIX_TS}"
    echo "[pix_ts] using PIX_TS=$PIX_TS (dim=$dim), patched config: $PATCHED_CONFIG"
fi

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results_modes.txt"

echo "" >> "$RESULTS_FILE"
echo "--- 12x1152 [1,3,5,7] null_s_clean @ ep${EP} : s_clean mode / cfg / decouple sweep ---" >> "$RESULTS_FILE"
echo "ckpt: $CKPT" >> "$RESULTS_FILE"
echo "config: $CONFIG  (PIX_TS override: ${PIX_TS:-config-default})" >> "$RESULTS_FILE"
printf "%-8s  %-8s  %-8s  %-8s  %-8s  %s\n" "mode" "pixcfg" "t_min" "t_max" "raecfg" "FID5k" >> "$RESULTS_FILE"
echo "------------------------------------------------------------" >> "$RESULTS_FILE"

# run_combo MODE PIX_CFG_SCALE T_MIN T_MAX [RAE_CFG_SCALE]
# 5th arg optional: empty → RAE-CFG coupled to pix scale (default);
#                   1.0   → RAE-CFG OFF (Phase 1 = single natural conditional latent).
# pixcfg=1.00 with any mode == no CFG at all (sampler skips the cond/uncond split).
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
    tag=$(printf "sclean-%s-s%.2f-tmin%.2f-tmax%.2f%s%s" "$mode" "$scale" "$t_min" "$t_max" "$rae_tag" "$PIX_TS_TAG")
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
    fid_out=$(cd /home/shihaoyang/RAE/src && "$PYTHON" -m eval.fid \
        --arr1 "$npz_path" \
        --ref "$REF" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "$fid_out" | tail -1)
    echo "  -> FID: $fid_val"
    printf "%-8s  %-8s  %-8s  %-8s  %-8s  %s\n" "$mode" "$scale" "$t_min" "$t_max" "$rae_col" "$fid_val" >> "$RESULTS_FILE"
}

if [ -n "$INTERVAL" ]; then
    # ===== CFG-interval sweep (null, cfg=2.0): restrict CFG to [t_min, t_max] =====
    # t=1 is NOISE, t=0 is CLEAN. (0.0,1.0) row = full-CFG anchor.
    for tmin in 0.0 0.1 0.2 0.3; do
        run_combo null 2.0 "$tmin" 1.0
    done
    run_combo null 2.0 0.0 0.8
    run_combo null 2.0 0.1 0.9
elif [ -n "$DECOUPLE" ]; then
    # ===== RAE <-> Pixel CFG decouple (rae-pixel-cfg-split), null mode =====
    # 5th arg = RAE-CFG scale, decoupled from the Pixel-CFG scale (2nd arg).
    # rae=1.0 → RAE-CFG OFF (Phase 1 = one natural conditional latent); pixel CFG still on.
    run_combo null 2.0 0.0 1.0 1.0      # rae-off anchor
    for rae in 1.5 2.0 2.5; do
        run_combo null 2.0 0.0 1.0 "$rae"
        run_combo null 2.5 0.0 1.0 "$rae"
    done
else
    # ===== DEFAULT block: lean first look at the chosen checkpoint (GPU-friendly) =====
    # (0) No-CFG baseline for comparison (pixcfg=1.00 → sampler runs a single conditional pass).
    run_combo null 1.0 0.0 1.0

    # (1) null-mode CFG-scale sweep (null = the mode this model was trained for).
    # run_combo null 1.5 0.0 1.0
    # run_combo null 2.0 0.0 1.0
    # run_combo null 2.5 0.0 1.0
    run_combo null 1.4 0.0 1.0
    run_combo null 1.5 0.0 1.0
    run_combo null 1.6 0.0 1.0

    # (2) 4-mode s_clean compare — DISABLED: prior experiments found shared/split/traj
    #     add little over plain null. Re-enable by uncommenting if you want the ablation back.
    # for mode in shared split traj; do
    #     run_combo "$mode" 2.0 0.0 1.0
    # done
    # run_combo null 2.0 0.0 1.0   # (null @ 2.0 already covered by block (1))
fi

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
