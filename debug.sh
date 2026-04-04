export CUDA_VISIBLE_DEVICES=4
export EXPERIMENT_NAME=dev
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    src/train_stage3.py \
    --config ~/RAE/configs/stage2/training/PixelDDTv2.yaml \
    --data-path /data/shihaoyang/data/imagenet/256/train \
    --results-dir /data/zhangchi/ckpts/PixDDTv2 \
    --precision fp32 \
    --compile \
    --wandb \

  # 256/train