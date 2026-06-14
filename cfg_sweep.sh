#!/bin/bash
# CFG-scale sweep (FID5k) for the 16x1536 layer-6 null_s_clean model.
# Uses sample_pix_ddt_project_free_layer.py and forces pix_ts=1 by default.
#
# Run from RAE/ root:
#   bash cfg_sweep.sh
#   CFG_SCALES="1.2 1.4 1.6 1.8" bash cfg_sweep.sh
#   NUM_FID_SAMPLES=50000 bash cfg_sweep.sh
#
# Env knobs: NPROC=<visible GPU count>, NUM_FID_SAMPLES=5000, BATCH_SIZE=64, PIX_TS=1.

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean}
EP=${EP:-0000050}
CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/checkpoints/ep-0000050.pt}
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
PYTHON=${PYTHON:-/home/shihaoyang/miniconda3/envs/rae/bin/python}
if [ -z "${NPROC:-}" ]; then
    NPROC=$("$PYTHON" -c "import torch; print(max(torch.cuda.device_count(), 1))")
fi
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-5000}
BATCH_SIZE=${BATCH_SIZE:-64}
PRECISION=${PRECISION:-fp32}

CFG_SCALES=${CFG_SCALES:-"1.0 1.2 1.5 2.0 2.5"}
T_MIN=${T_MIN:-0.0}
T_MAX=${T_MAX:-1.0}
S_CLEAN_CFG_MODE=${S_CLEAN_CFG_MODE:-null}

# Force cfg-sweep sampling to pix_ts=1 unless explicitly overridden.
# pix_ts maps to misc.time_dist_shift_dim = round(PIX_TS^2 * 4096).
PIX_TS=${PIX_TS:-1}
SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXPERIMENT_NAME}/cfg_sweep_ep${EP}_pixts${PIX_TS}}

dim=$("$PYTHON" -c "print(round($PIX_TS**2*4096))")
PATCHED_CONFIG=/tmp/cfg_sweep_1536_ep${EP}_pixts${PIX_TS}.yaml
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
echo "[pix_ts] using PIX_TS=$PIX_TS (dim=$dim), patched config: $PATCHED_CONFIG"

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results_cfg_scale.txt"

echo "" >> "$RESULTS_FILE"
echo "--- 16x1536 layer-6 null_s_clean @ ep${EP} : cfg_scale sweep ---" >> "$RESULTS_FILE"
echo "ckpt: $CKPT" >> "$RESULTS_FILE"
echo "config: $CONFIG" >> "$RESULTS_FILE"
echo "pix_ts: $PIX_TS" >> "$RESULTS_FILE"
echo "nproc: $NPROC" >> "$RESULTS_FILE"
printf "%-8s  %-8s  %-8s  %-8s  %s\n" "mode" "cfg" "t_min" "t_max" "FID5k" >> "$RESULTS_FILE"
echo "------------------------------------------------" >> "$RESULTS_FILE"

run_combo() {
    local scale=$1

    local tag
    tag=$(printf "sclean-%s-s%.2f-tmin%.2f-tmax%.2f%s" "$S_CLEAN_CFG_MODE" "$scale" "$T_MIN" "$T_MAX" "$PIX_TS_TAG")
    local npz_path="$SAMPLE_BASE/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  mode=$S_CLEAN_CFG_MODE  cfg=$scale  t_min=$T_MIN  t_max=$T_MAX  pix_ts=$PIX_TS"
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
            --t-min "$T_MIN" \
            --t-max "$T_MAX" \
            --s-clean-cfg-mode "$S_CLEAN_CFG_MODE"
    fi

    echo "  Computing FID..."
    fid_out=$(cd /home/shihaoyang/RAE/src && "$PYTHON" -m eval.fid \
        --arr1 "$npz_path" \
        --ref "$REF" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "$fid_out" | tail -1)
    echo "  -> FID: $fid_val"
    printf "%-8s  %-8s  %-8s  %-8s  %s\n" "$S_CLEAN_CFG_MODE" "$scale" "$T_MIN" "$T_MAX" "$fid_val" >> "$RESULTS_FILE"
}

for scale in $CFG_SCALES; do
    run_combo "$scale"
done

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
