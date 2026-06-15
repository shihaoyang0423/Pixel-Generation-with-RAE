#!/usr/bin/env bash
# AutoGuidance scale sweep for 12x1152 with DiTDH-S_ep14 as the bad DiTDH.
#
# Good PixelDDT: 12x1152 Pixel Head checkpoint passed via --ckpt.
# Good DiTDH  : DiTDH-XL, matching the original 12x1152 config.
# Bad DiTDH   : DiTDH-S_ep14, instantiated with explicit DiTDH-S params.
# Pixel time shift defaults to pix_ts=1 and can be overridden with PIX_TS=<value>.

set -euo pipefail

EXP_NAME="${EXP_NAME:-PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDH-XL}"
CKPT_DIR="${CKPT_DIR:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXP_NAME}/checkpoints}"

PIXEL_CKPT="${PIXEL_CKPT:-${CKPT_DIR}/ep-0000050.pt}"
CONFIG="${CONFIG:-/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_12x1152_project_free_layer_1_3_5_7_DiTDHXL_pix_ts1_AG_bad_DiTDHS.yaml}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/home/shihaoyang/data/pretrained/DiTs/Dinov2/wReg_base/ImageNet256}"
REF="${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}"
PYTHON="${PYTHON:-/home/shihaoyang/miniconda3/envs/rae/bin/python}"
PIX_TS="${PIX_TS:-1}"
SAMPLE_BASE="${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXP_NAME}_autoguidance_ditdh_condition_bad_s_ep14_pixts${PIX_TS}}"

GOOD_DITDH_CKPT="${PRETRAIN_DIR}/DiTDH-XL/stage2_model.pt"
BAD_DITDH_CKPT="${PRETRAIN_DIR}/DiTDH-S_ep14/stage2_model.pt"

SCALES=(${SCALES:-1.00 1.50 2.00 2.50 3.00 3.50 4.00})

NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-5000}"
PRECISION="${PRECISION:-fp32}"

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] config not found: ${CONFIG}" >&2
    exit 1
fi

PIX_TS_DIM=$("${PYTHON}" -c "print(round(float('${PIX_TS}')**2 * 4096))")
PATCHED_CONFIG="/tmp/ag_sweep_1152_s_bad_pixts${PIX_TS}.yaml"
"${PYTHON}" - "${CONFIG}" "${PATCHED_CONFIG}" "${PIX_TS_DIM}" <<'PY'
import sys
import yaml

src, dst, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("misc", {})["time_dist_shift_dim"] = dim
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY
CONFIG="${PATCHED_CONFIG}"
PIX_TS_TAG="-pixts${PIX_TS}"

mkdir -p "${SAMPLE_BASE}"
RESULTS_FILE="${SAMPLE_BASE}/fid_results.txt"

if [ ! -f "${PIXEL_CKPT}" ]; then
    echo "[ERROR] 12x1152 PixelDDT checkpoint not found: ${PIXEL_CKPT}" >&2
    echo "        Set PIXEL_CKPT=/path/to/12x1152/checkpoints/ep-*.pt and rerun." >&2
    exit 1
fi

if [ ! -f "${GOOD_DITDH_CKPT}" ]; then
    echo "[ERROR] good DiTDH-XL checkpoint not found: ${GOOD_DITDH_CKPT}" >&2
    exit 1
fi

if [ ! -f "${BAD_DITDH_CKPT}" ]; then
    echo "[ERROR] bad DiTDH-S checkpoint not found: ${BAD_DITDH_CKPT}" >&2
    exit 1
fi

{
    echo ""
    echo "--- 12x1152 DiTDH-condition AutoGuidance sweep with bad DiTDH-S_ep14: ${EXP_NAME} ---"
    echo "pixel_ckpt=${PIXEL_CKPT}"
    echo "config=${CONFIG}"
    echo "pretrain_dir=${PRETRAIN_DIR}"
    echo "good_ditdh=${GOOD_DITDH_CKPT}"
    echo "bad_ditdh=${BAD_DITDH_CKPT}"
    echo "pix_ts=${PIX_TS} (time_dist_shift_dim=${PIX_TS_DIM}, base=4096)"
    printf "%-18s  %-18s  %-8s  %-8s  %-8s  %s\n" "role" "guidance_ditdh" "scale" "t_min" "t_max" "FID5k"
    echo "-------------------------------------------------------"
} >> "${RESULTS_FILE}"

run_combo() {
    local scale="$1"
    local t_min="$2"
    local t_max="$3"
    local bad_name
    bad_name="$(basename "$(dirname "${BAD_DITDH_CKPT}")")"
    local tag
    tag=$(printf "ag1152-bad_DiTDH-S_ep14-%s-s%.2f-tmin%.1f-tmax%.1f%s" "${bad_name}" "${scale}" "${t_min}" "${t_max}" "${PIX_TS_TAG}")
    local npz_path="${SAMPLE_BASE}/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  12x1152 AutoGuidance role=bad_DiTDH-S_ep14 guidance_ditdh=${bad_name} scale=${scale} t_min=${t_min} t_max=${t_max} pix_ts=${PIX_TS}"
    echo "  tag=${tag}"
    echo "============================================================"

    if [ -f "${npz_path}" ]; then
        echo "  [SKIP] npz already exists: ${npz_path}"
    else
        SAVE_FOLDER="${tag}" \
        torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
            src/sample_pix_ddt_project_free_layer.py \
            --config "${CONFIG}" \
            --ckpt "${PIXEL_CKPT}" \
            --ag-ckpt "${BAD_DITDH_CKPT}" \
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
    fid_out=$(cd /home/shihaoyang/RAE/src && "${PYTHON}" -m eval.fid \
        --arr1 "${npz_path}" \
        --ref "${REF}" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    fid_val=$(echo "${fid_out}" | tail -1)
    echo "  -> FID: ${fid_val}"
    printf "%-18s  %-18s  %-8s  %-8s  %-8s  %s\n" "bad_DiTDH-S_ep14" "${bad_name}" "${scale}" "${t_min}" "${t_max}" "${fid_val}" >> "${RESULTS_FILE}"
}

for scale in "${SCALES[@]}"; do
    run_combo "${scale}" 0.0 1.0
done

echo ""
echo "========================================"
echo "ALL DONE. Results appended to: ${RESULTS_FILE}"
echo "========================================"
cat "${RESULTS_FILE}"
