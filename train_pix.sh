torchrun --standalone --nnodes=1 --nproc_per_node=6 \
  src/train_stage3.py \
  --config /home/zhangchi/RAE/configs/stage2/training/PixelDDT.yaml \
  --data-path /data/zhangchi/data/imagenet/256/train \
  --results-dir /data/zhangchi/ckpts/PixDDT \
  --precision bf16 \
  --compile \
  --wandb

  # 256/train