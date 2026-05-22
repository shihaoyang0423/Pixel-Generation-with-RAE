#!/usr/bin/env bash
# 训练崩溃自动重启脚本 — p2nodrop 实验版本。
#
# 与 retry_train_stage3_project_free_layer.sh 的唯一实质区别：
#   · TRAIN_CONFIG → PixelDDTv2_small_lora_project_free_layer_p2nodrop.yaml
#   · SAMPLE_CONFIG → PixDDT_small_v2_lora_project_free_layer_p2nodrop_cfg_sweep.yaml
#   · 快照额外拷贝 PixelDDTv2_project_free_layer_p2nodrop.py（p2nodrop 模型文件）
#   · 默认 EXPERIMENT_NAME 已改
#
# bootstrap 逻辑：现有 retry_snap_bootstrap.py 已将
#   stage2.models.PixelDDTv2_project_free_layer（基类）注册进 sys.modules。
# p2nodrop 子类在被 instantiate_from_config 动态加载时，通过相对导入自动
# 拿到上面那个已快照的基类版本，无需额外注册。
#
# 退出码约定（由 train_stage3_projcet_free_layer.py 保证）：
#   0  → 训练正常完成
#   1  → NaN loss，直接重启
#   2  → ckpt 文件损坏，删掉最新 ckpt 再重启
#   其他 → 其他崩溃（OOM / wandb 超时等），直接重启
#
# 用法（tmux 内）：
#   tmux new -s train_p2nodrop
#   export EXPERIMENT_NAME=PixDDT_small_v2_project_free_layer_p2nodrop
#   bash /home/shihaoyang/RAE/retry_train_stage3_project_free_layer_p2nodrop.sh
#   Ctrl-B D  → 退出 tmux 但脚本继续跑
#   Ctrl-C    → 停止脚本

export CUDA_VISIBLE_DEVICES=4,5,6,7  # 记得 check

export EXPERIMENT_NAME="PixDDT_small_v2_project_free_layer_p2nodrop"  # 记得 check

CKPT_DIR="/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints"

cd /home/shihaoyang/RAE
RAE_ROOT="/home/shihaoyang/RAE"
RAE_SRC="${RAE_ROOT}/src"

# 启动时快照 YAML + 关键 Python；本脚本整段生命周期只用副本。
_SNAP_DIR=$(mktemp -d /tmp/retry_train_stage3_p2nodrop.XXXXXX)
TRAIN_CONFIG="${_SNAP_DIR}/PixelDDTv2_small_lora_project_free_layer_p2nodrop.yaml"
SAMPLE_CONFIG="${_SNAP_DIR}/PixDDT_small_v2_lora_project_free_layer_p2nodrop_cfg_sweep.yaml"
cp -- "${RAE_ROOT}/configs/stage2/training/PixelDDTv2_small_lora_project_free_layer_p2nodrop.yaml" "${TRAIN_CONFIG}"
cp -- "${RAE_ROOT}/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer_p2nodrop_cfg_sweep.yaml" "${SAMPLE_CONFIG}"

_SNAPSHOT_SRC="${_SNAP_DIR}/src"
mkdir -p "${_SNAPSHOT_SRC}"
cp -- "${RAE_SRC}/train_stage3_projcet_free_layer.py"                            "${_SNAPSHOT_SRC}/"
cp -- "${RAE_SRC}/sample_pix_ddt_project_free_layer.py"                          "${_SNAPSHOT_SRC}/"
cp -- "${RAE_SRC}/stage2/models/PixelDDTv2_project_free_layer.py"                "${_SNAPSHOT_SRC}/PixelDDTv2_project_free_layer.py"
cp -- "${RAE_SRC}/stage2/models/PixelDDTv2_project_free_layer_p2nodrop.py"       "${_SNAPSHOT_SRC}/PixelDDTv2_project_free_layer_p2nodrop.py"
cp -- "${RAE_ROOT}/scripts/retry_snap_bootstrap.py"                              "${_SNAPSHOT_SRC}/_retry_snap_bootstrap.py"
BOOTSTRAP_SCRIPT="${_SNAPSHOT_SRC}/_retry_snap_bootstrap.py"

export RAE_SRC RAE_ROOT
export PYTHONPATH="${RAE_SRC}:${RAE_ROOT}:${PYTHONPATH:-}"
echo "[snap] config+code frozen at: ${_SNAP_DIR}"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)

SAMPLE_DIR="${HOME}/data/samples/${EXPERIMENT_NAME}"
FID_RECORD="/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/fid_record.txt"
REF_NPZ="/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz"
TRAIN_TEE_LOG="/home/shihaoyang/RAE/train_project_free_layer_${EXPERIMENT_NAME}.log"
FID_TEE_LOG="/home/shihaoyang/RAE/train_project_free_layer_fid_${EXPERIMENT_NAME}.log"

while true; do
    RETRY_SNAP_MODE=train torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        "${BOOTSTRAP_SCRIPT}" \
        --config "${TRAIN_CONFIG}" \
        --data-path /data/shihaoyang/data/imagenet/256/train \
        --results-dir /home/shihaoyang/data/ckpts/PixDDTv2 \
        --precision fp32 \
        --compile \
        --wandb \
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
    fi

    echo "[retry] rc=${rc}, restarting in 30s... ($(date '+%Y-%m-%d %H:%M:%S'))"
    python3 -c "import torch, gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true
    sleep 30
done
