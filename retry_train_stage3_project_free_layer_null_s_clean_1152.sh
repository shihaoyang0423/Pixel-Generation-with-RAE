#!/usr/bin/env bash
# Train free_layer PixelDDT (large 12x1152 / DiTDH-XL) from scratch with
# synchronized y / null_s_clean CFG dropout.
#   - depth=12, hidden=1152, num_heads=16, rae_inject_layers=[1, 3, 5, 7]
#   - use_s_clean_null_token=true
#   - trained from scratch (only the frozen DiTDH-XL backbone is loaded; no --ckpt)

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean}

TRAIN_CONFIG_SRC=${TRAIN_CONFIG_SRC:-/home/shihaoyang/RAE/configs/stage2/training/PixelDDT_free_layer_1_3_5_7_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean.yaml}
SAMPLE_CONFIG_SRC=${SAMPLE_CONFIG_SRC:-/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_free_layer_1_3_5_7_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDHXL_null_s_clean_pix_ts3.yaml}

CKPT_DIR="/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints"

cd /home/shihaoyang/RAE
RAE_ROOT="/home/shihaoyang/RAE"
RAE_SRC="${RAE_ROOT}/src"

_SNAP_DIR=$(mktemp -d /tmp/retry_train_stage3_proj_fl_null_s_1152.XXXXXX)
TRAIN_CONFIG="${_SNAP_DIR}/$(basename "${TRAIN_CONFIG_SRC}")"
SAMPLE_CONFIG="${_SNAP_DIR}/$(basename "${SAMPLE_CONFIG_SRC}")"
cp -- "${TRAIN_CONFIG_SRC}" "${TRAIN_CONFIG}"
cp -- "${SAMPLE_CONFIG_SRC}" "${SAMPLE_CONFIG}"

_SNAPSHOT_SRC="${_SNAP_DIR}/src"
mkdir -p "${_SNAPSHOT_SRC}"
cp -- "${RAE_SRC}/train_stage3_projcet_free_layer.py" "${_SNAPSHOT_SRC}/"
cp -- "${RAE_SRC}/sample_pix_ddt_project_free_layer.py" "${_SNAPSHOT_SRC}/"
cp -- "${RAE_SRC}/stage2/models/PixelDDTv2_project_free_layer.py" "${_SNAPSHOT_SRC}/PixelDDTv2_project_free_layer.py"
cp -- "${RAE_ROOT}/scripts/retry_snap_bootstrap.py" "${_SNAPSHOT_SRC}/_retry_snap_bootstrap.py"
BOOTSTRAP_SCRIPT="${_SNAPSHOT_SRC}/_retry_snap_bootstrap.py"

export RAE_SRC RAE_ROOT
export PYTHONPATH="${RAE_SRC}:${RAE_ROOT}:${PYTHONPATH:-}"
echo "[snap] config+code frozen at: ${_SNAP_DIR}"
echo "[init] from scratch (no --ckpt; only frozen DiTDH-XL backbone loaded)"
echo "[exp] EXPERIMENT_NAME=${EXPERIMENT_NAME}"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
WANDB_ARGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    if [ -n "${ENTITY:-}" ] && [ -n "${PROJECT:-}" ]; then
        WANDB_ARGS=(--wandb)
        echo "[wandb] enabled: ENTITY=${ENTITY}, PROJECT=${PROJECT}"
    else
        echo "[wandb] disabled because ENTITY or PROJECT is not set. Set USE_WANDB=0 to silence this."
    fi
else
    echo "[wandb] disabled by USE_WANDB=${USE_WANDB}"
fi

COMPILE_ARGS=()
if [ "${USE_COMPILE:-1}" = "1" ]; then
    COMPILE_ARGS=(--compile)
    echo "[compile] enabled by USE_COMPILE=1"
else
    echo "[compile] disabled by USE_COMPILE=${USE_COMPILE}"
fi

SAMPLE_DIR="${HOME}/data/samples/${EXPERIMENT_NAME}"
FID_RECORD="/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/fid_record.txt"
REF_NPZ="/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz"
TRAIN_TEE_LOG="/home/shihaoyang/RAE/train_project_free_layer_${EXPERIMENT_NAME}.log"
FID_TEE_LOG="/home/shihaoyang/RAE/train_project_free_layer_fid_${EXPERIMENT_NAME}.log"

while true; do
    RETRY_SNAP_MODE=train torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
        "${BOOTSTRAP_SCRIPT}" \
        --config "${TRAIN_CONFIG}" \
        --data-path /data/shihaoyang/data/imagenet/256/train \
        --results-dir /home/shihaoyang/data/ckpts/PixDDTv2 \
        --precision fp32 \
        "${COMPILE_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        2>&1 | tee -a "${TRAIN_TEE_LOG}"

    rc=${PIPESTATUS[0]}

    fid_signal=$(ls "${CKPT_DIR}"/ep-*.fid_needed 2>/dev/null | sort -V | tail -n1)
    if [ -n "${fid_signal}" ]; then
        ckpt_base=$(basename "${fid_signal}" .fid_needed)
        latest_10="${CKPT_DIR}/${ckpt_base}.pt"
        marker="${CKPT_DIR}/${ckpt_base}.fid_done"
        echo "[fid] Signal detected, evaluating checkpoint: ${latest_10}"
        RETRY_SNAP_MODE=sample torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
            "${BOOTSTRAP_SCRIPT}" \
            --config "${SAMPLE_CONFIG}" \
            --ckpt "${latest_10}" \
            --sample-dir "${SAMPLE_DIR}" \
            --num-fid-samples 5000 \
            --per-proc-batch-size 64 \
            --precision fp32 \
            --label-sampling equal \
            --naive \
            --cfg-scale 1.50 \
            --t-min 0.0 \
            --t-max 1.0 \
            --s-clean-cfg-mode null \
            2>&1 | tee -a "${FID_TEE_LOG}"
        npz_file=$(ls -t "${SAMPLE_DIR}"/*.npz 2>/dev/null | head -1)
        if [ -n "${npz_file}" ]; then
            base="${npz_file%.npz}"
            npz_fid5k="${base}-fid5k.npz"
            mv "${npz_file}" "${npz_fid5k}"
            [ -d "${base}" ] && mv "${base}" "${base}-fid5k"
            echo "[fid] Computing FID from: ${npz_fid5k}"
            fid_result=$(PYTHONPATH=/home/shihaoyang/RAE/src /home/shihaoyang/miniconda3/envs/rae/bin/python -m eval.fid \
                --arr1 "${npz_fid5k}" \
                --ref "${REF_NPZ}" \
                --ref-mu-key mu \
                --ref-sigma-key sigma 2>&1)
            echo "[fid] Result for ${ckpt_base}: ${fid_result}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${ckpt_base}: ${fid_result}" >> "${FID_RECORD}"
        else
            echo "[fid] No npz found in ${SAMPLE_DIR}, FID skipped."
        fi
        rm -f "${fid_signal}"
        touch "${marker}"
    fi

    if [ $rc -eq 0 ]; then
        echo "[retry] Training finished successfully."
        break
    elif [ $rc -eq 2 ]; then
        latest=$(ls "${CKPT_DIR}"/ep-[0-9]*.pt 2>/dev/null | sort | tail -n1)
        if [ -n "$latest" ]; then
            echo "[retry] Corrupt checkpoint, deleting: $latest"
            rm -f "$latest"
        fi
    else
        echo "[retry] Training exited with code ${rc}; retrying in 30s."
    fi
    sleep 30
done
