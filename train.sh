export CUDA_VISIBLE_DEVICES=2
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/train.py \
  --config /home/zhangchi/RAE/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path /data/zhangchi/data/imagenet/256/train \
  --results-dir /data/zhangchi/ckpts/RAE-DDT-baseline \
  --precision bf16 \
  --compile \
  --wandb