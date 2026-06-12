#!/usr/bin/env bash
# AutoGuidance scale sweep for the 12x1152 project_free_layer model.
#
# Good PixelDDT: 12x1152 Pixel Head checkpoint passed via --ckpt.
# Good DiTDH  : DiTDH-XL, matching the original 12x1152 config.
# Bad DiTDH   : weaker DiTDH-XL checkpoints by default (ep20, ep80).
# Pixel time shift defaults to pix_ts=1 via the dedicated AG YAML.

set -euo pipefail

EXP_NAME="${EXP_NAME:-PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDH-XL}"
CKPT_DIR="${CKPT_DIR:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXP_NAME}/checkpoints}"

PIXEL_CKPT="${PIXEL_CKPT:-${CKPT_DIR}/ep-0000050.pt}"
CONFIG="${CONFIG:-/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_12x1152_project_free_layer_1_3_5_7_DiTDHXL_pix_ts1_AG.yaml}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/home/shihaoyang/data/pretrained/DiTs/Dinov2/wReg_base/ImageNet256}"
SAMPLE_BASE="${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXP_NAME}_autoguidance_ditdh_condition}"
REF="${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}"

GOOD_DITDH_CKPT="${PRETRAIN_DIR}/DiTDH-XL/stage2_model.pt"
BAD_DITDH_CKPTS=(
    "bad_DiTDH-XL_ep20:${PRETRAIN_DIR}/DiTDH-XL_ep20/stage2_model.pt"
    "bad_DiTDH-XL_ep80:${PRETRAIN_DIR}/DiTDH-XL_ep80/stage2_model.pt"
)

SCALES=(${SCALES:-1.00 1.50 2.00 2.50 3.00 3.50 4.00})

NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-5000}"
PRECISION="${PRECISION:-fp32}"

mkdir -p "${SAMPLE_BASE}"
RESULTS_FILE="${SAMPLE_BASE}/fid_results.txt"

if [ ! -f "${PIXEL_CKPT}" ]; then
    echo "[ERROR] 12x1152 PixelDDT checkpoint not found: ${PIXEL_CKPT}" >&2
    echo "        Set PIXEL_CKPT=/path/to/12x1152/checkpoints/ep-*.pt and rerun." >&2
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] config not found: ${CONFIG}" >&2
    exit 1
fi

if [ ! -f "${GOOD_DITDH_CKPT}" ]; then
    echo "[ERROR] good DiTDH-XL checkpoint not found: ${GOOD_DITDH_CKPT}" >&2
    exit 1
fi

{
    echo ""
    echo "--- 12x1152 DiTDH-condition AutoGuidance sweep: ${EXP_NAME} ---"
    echo "pixel_ckpt=${PIXEL_CKPT}"
    echo "config=${CONFIG}"
    echo "pretrain_dir=${PRETRAIN_DIR}"
    echo "good_ditdh=${GOOD_DITDH_CKPT}"
    echo "pix_ts=1.0 (time_dist_shift_dim=4096, base=4096)"
    printf "%-18s  %-18s  %-8s  %-8s  %-8s  %s\n" "role" "guidance_ditdh" "scale" "t_min" "t_max" "FID5k"
    echo "-------------------------------------------------------"
} >> "${RESULTS_FILE}"

run_combo() {
    local role="$1"
    local guidance_ckpt="$2"
    local scale="$3"
    local t_min="$4"
    local t_max="$5"
    local bad_name
    bad_name="$(basename "$(dirname "${guidance_ckpt}")")"
    local tag
    tag=$(printf "ag1152-%s-%s-s%.2f-tmin%.1f-tmax%.1f-pixts1" "${role}" "${bad_name}" "${scale}" "${t_min}" "${t_max}")
    local npz_path="${SAMPLE_BASE}/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  12x1152 AutoGuidance role=${role} guidance_ditdh=${bad_name} scale=${scale} t_min=${t_min} t_max=${t_max}"
    echo "  tag=${tag}"
    echo "============================================================"

    if [ ! -f "${guidance_ckpt}" ]; then
        echo "  [SKIP] bad DiTDH checkpoint not found: ${guidance_ckpt}" >&2
        return
    fi

    if [ -f "${npz_path}" ]; then
        echo "  [SKIP] npz already exists: ${npz_path}"
    else
        SAVE_FOLDER="${tag}" \
        torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
            src/sample_pix_ddt_project_free_layer.py \
            --config "${CONFIG}" \
            --ckpt "${PIXEL_CKPT}" \
            --ag-ckpt "${guidance_ckpt}" \
            --guidance-method autoguidance \
            --ag-phase condition \
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
    printf "%-18s  %-18s  %-8s  %-8s  %-8s  %s\n" "${role}" "${bad_name}" "${scale}" "${t_min}" "${t_max}" "${fid_val}" >> "${RESULTS_FILE}"
}

for item in "${BAD_DITDH_CKPTS[@]}"; do
    role="${item%%:*}"
    ckpt="${item#*:}"
    for scale in "${SCALES[@]}"; do
        run_combo "${role}" "${ckpt}" "${scale}" 0.0 1.0
    done
done

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: ${RESULTS_FILE}"
echo "========================================"
cat "${RESULTS_FILE}"
