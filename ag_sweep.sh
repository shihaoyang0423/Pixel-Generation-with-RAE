#!/usr/bin/env bash
# AutoGuidance scale sweep for the 6x768 project_free_layer model.
#
# Good model: ep-0000050.pt
# Bad model : ep-0000020.pt (same architecture, weaker checkpoint)
# Pixel time shift: pix_ts=1.0, from YAML misc.time_dist_shift_dim/base = 4096/4096.
#
# Run from RAE repo root:
#   bash ag_sweep.sh

set -euo pipefail

EXP_NAME="PixDDT_small_v2_project_free_layer_2_false_logit_normal_0.8_0.8_pix_ts_1_wo_lora_w_y_embed_adaln_zero"
CKPT_DIR="/home/shihaoyang/data/ckpts/PixDDTv2/${EXP_NAME}/checkpoints"

GOOD_CKPT="${CKPT_DIR}/ep-0000050.pt"
BAD_CKPT="${CKPT_DIR}/ep-0000020.pt"
CONFIG="/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_6x768_project_free_layer_2_false_pix_ts1_AG.yaml"
SAMPLE_BASE="/home/shihaoyang/data/samples/${EXP_NAME}_autoguidance_ep50_good_ep20_bad_pix_ts1"
REF="/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz"

NPROC="${NPROC:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-5000}"
PRECISION="${PRECISION:-fp32}"

mkdir -p "${SAMPLE_BASE}"
RESULTS_FILE="${SAMPLE_BASE}/fid_results.txt"

if [ ! -f "${GOOD_CKPT}" ]; then
    echo "[ERROR] good checkpoint not found: ${GOOD_CKPT}" >&2
    exit 1
fi

if [ ! -f "${BAD_CKPT}" ]; then
    echo "[ERROR] bad checkpoint not found: ${BAD_CKPT}" >&2
    echo "        This sweep is configured to use ep-0000020.pt as the bad model." >&2
    echo "        If ep20 was deleted, point BAD_CKPT to another early checkpoint." >&2
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] config not found: ${CONFIG}" >&2
    exit 1
fi

{
    echo ""
    echo "--- AutoGuidance sweep: ${EXP_NAME} ---"
    echo "good=${GOOD_CKPT}"
    echo "bad=${BAD_CKPT}"
    echo "config=${CONFIG}"
    echo "pix_ts=1.0 (time_dist_shift_dim=4096, base=4096)"
    printf "%-8s  %-8s  %-8s  %s\n" "scale" "t_min" "t_max" "FID5k"
    echo "----------------------------------------"
} >> "${RESULTS_FILE}"

run_combo() {
    local scale="$1"
    local t_min="$2"
    local t_max="$3"
    local tag
    tag=$(printf "ag-s%.2f-tmin%.1f-tmax%.1f-pix_ts1" "${scale}" "${t_min}" "${t_max}")
    local npz_path="${SAMPLE_BASE}/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  AutoGuidance scale=${scale} t_min=${t_min} t_max=${t_max}"
    echo "  tag=${tag}"
    echo "============================================================"

    if [ -f "${npz_path}" ]; then
        echo "  [SKIP] npz already exists: ${npz_path}"
    else
        SAVE_FOLDER="${tag}" \
        torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
            src/sample_pix_ddt_project_free_layer.py \
            --config "${CONFIG}" \
            --ckpt "${GOOD_CKPT}" \
            --ag-ckpt "${BAD_CKPT}" \
            --guidance-method autoguidance \
            --ag-phase pixel \
            --sample-dir "${SAMPLE_BASE}" \
            --num-fid-samples "${NUM_FID_SAMPLES}" \
            --per-proc-batch-size "${BATCH_SIZE}" \
            --precision "${PRECISION}" \
            --label-sampling equal \
            --naive \
            --cfg-scale "${scale}" \
            --t-min "${t_min}" \
            --t-max "${t_max}"
    fi

    echo "  Computing FID..."
    fid_out=$(cd /home/shihaoyang/RAE/src && python -m eval.fid \
        --arr1 "${npz_path}" \
        --ref "${REF}" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "${fid_out}" | tail -1)
    echo "  -> FID: ${fid_val}"
    printf "%-8s  %-8s  %-8s  %s\n" "${scale}" "${t_min}" "${t_max}" "${fid_val}" >> "${RESULTS_FILE}"
}

# AutoGuidance scales are not CFG scales. scale=1.00 is the good-only control:
# bad + 1.00 * (good - bad) = good.
run_combo 1.00 0.0 1.0
run_combo 1.02 0.0 1.0
run_combo 1.05 0.0 1.0
run_combo 1.10 0.0 1.0
run_combo 1.20 0.0 1.0

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: ${RESULTS_FILE}"
echo "========================================"
cat "${RESULTS_FILE}"
