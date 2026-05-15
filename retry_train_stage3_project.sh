#!/usr/bin/env bash
# 训练崩溃自动重启脚本 (project 版本 — 串行去噪)。
# 退出码约定（由 train_stage3_projcet.py 保证）：
#   0  → 训练正常完成
#   1  → NaN loss，直接重启（从最新 ckpt 继续）
#   2  → ckpt 文件损坏（save 被中断），删掉最新 ckpt 再重启
#   其他→ 其他崩溃（OOM / wandb 超时 / compile 失败等），直接重启
#
# 用法（tmux 内）：
#   tmux new -s train_proj
#   export EXPERIMENT_NAME=PixDDT_small_v2_project
#   bash /home/shihaoyang/RAE/retry_train_stage3_project.sh
#   Ctrl-B D  → 退出 tmux 但脚本继续跑
#   Ctrl-C    → 停止脚本

export EXPERIMENT_NAME="PixDDT_small_v2_project_lora_16_0.01_PixelDDT_1152_16"  # 记得check

CKPT_DIR="/data/shihaoyang/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints"

cd /home/shihaoyang/RAE

export CUDA_VISIBLE_DEVICES=0,1,2,3  # 记得check
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)

SAMPLE_DIR="${HOME}/data/samples/${EXPERIMENT_NAME}"
FID_RECORD="/data/shihaoyang/ckpts/PixDDTv2/${EXPERIMENT_NAME}/fid_record.txt"
REF_NPZ="/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz"
SAMPLE_CONFIG="/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project.yaml"

while true; do
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        src/train_stage3_projcet.py \
        --config /home/shihaoyang/RAE/configs/stage2/training/PixelDDTv2_small_lora_project.yaml \
        --data-path /data/shihaoyang/data/imagenet/256/train \
        --results-dir /data/shihaoyang/ckpts/PixDDTv2 \
        --precision fp32 \
        --compile \
        --wandb \
        2>&1 | tee -a /home/shihaoyang/RAE/train_project_3.log  # 记得check

    rc=${PIPESTATUS[0]}

    # FID evaluation: triggered by .fid_needed signal file written by training script.
    # torchrun does NOT propagate child exit codes, so we use file-based signaling instead.
    fid_signal=$(ls "${CKPT_DIR}"/ep-*.fid_needed 2>/dev/null | sort -V | tail -n1)
    if [ -n "${fid_signal}" ]; then
        ckpt_base=$(basename "${fid_signal}" .fid_needed)
        latest_10="${CKPT_DIR}/${ckpt_base}.pt"
        marker="${CKPT_DIR}/${ckpt_base}.fid_done"
        echo "[fid] Signal detected, evaluating checkpoint: ${latest_10}"
        torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
            src/sample_pix_ddt_project.py \
            --config "${SAMPLE_CONFIG}" \
            --ckpt "${latest_10}" \
            --sample-dir "${SAMPLE_DIR}" \
            --num-fid-samples 5000 \
            --per-proc-batch-size 64 \
            --precision fp32 \
            --label-sampling equal \
            --naive \
            2>&1 | tee -a /home/shihaoyang/RAE/train_project_2.log
        npz_file=$(ls -t "${SAMPLE_DIR}"/*.npz 2>/dev/null | head -1)
        if [ -n "${npz_file}" ]; then
            # Rename npz and PNG dir to include fid5k tag
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
