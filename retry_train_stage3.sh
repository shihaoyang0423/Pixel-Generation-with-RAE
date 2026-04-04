#!/usr/bin/env bash
# 训练崩溃自动重启脚本。
# 退出码约定（由 train_stage3.py 保证）：
#   0  → 训练正常完成
#   1  → NaN loss，直接重启（从最新 ckpt 继续）
#   2  → ckpt 文件损坏（save 被中断），删掉最新 ckpt 再重启
#   其他→ 其他崩溃（OOM / wandb 超时 / compile 失败等），直接重启
#
# 用法（tmux 内）：
#   tmux new -s train
#   export EXPERIMENT_NAME=PixDDTv2_4
#   bash /home/shihaoyang/RAE/retry_train_stage3.sh
#   Ctrl-B D  → 退出 tmux 但脚本继续跑
#   Ctrl-C    → 停止脚本

export EXPERIMENT_NAME="PixDDT_small_v2mask"

CKPT_DIR="/data/shihaoyang/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints"

cd /home/shihaoyang/RAE

export CUDA_VISIBLE_DEVICES=0,1,2,3

while true; do
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        src/train_stage3.py \
        --config /home/shihaoyang/RAE/configs/stage2/training/PixelDDTv2.yaml \
        --data-path /data/shihaoyang/data/imagenet/256/train \
        --results-dir /data/shihaoyang/ckpts/PixDDTv2 \
        --precision fp32 \
        --compile \
        --wandb \
        2>&1 | tee -a /home/shihaoyang/RAE/train_full.log

    rc=${PIPESTATUS[0]}

    if [ $rc -eq 0 ]; then
        echo "[retry] Training finished successfully."
        break
    elif [ $rc -eq 2 ]; then
        # ckpt 文件损坏：删掉最新的（保证只删这一次，因为下次不会再是 rc=2）
        latest=$(ls "${CKPT_DIR}"/ep-[0-9]*.pt 2>/dev/null | sort | tail -n1)
        if [ -n "$latest" ]; then
            echo "[retry] Corrupt checkpoint, deleting: $latest"
            rm -f "$latest"
        fi
    fi

    echo "[retry] rc=${rc}, restarting in 30s... ($(date '+%Y-%m-%d %H:%M:%S'))"
    # 清理显存，等 GPU 释放
    python3 -c "import torch, gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true
    sleep 30
done
