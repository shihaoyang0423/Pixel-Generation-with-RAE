export CUDA_VISIBLE_DEVICES=4,5,6,7
export EXPERIMENT_NAME=dev
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/train_stage3.py \
    --config ~/RAE/configs/stage2/training/PixelDDTv2.yaml \
    --data-path /data/shihaoyang/data/imagenet/256/train \
    --results-dir /data/zhangchi/ckpts/PixDDTv2 \
    --precision fp32 \
    --compile \
    --wandb \

  # 256/train