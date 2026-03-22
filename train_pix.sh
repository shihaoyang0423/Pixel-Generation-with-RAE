export EXPERIMENT_NAME=x0_pred_768_dual_schedule_formal
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_stage3.py \
  --config /home/zhangchi/RAE/configs/stage2/training/x0_pred_768.yaml \
  --data-path /data/zhangchi/data/imagenet/256/train \
  --results-dir /data/zhangchi/ckpts/PixDDT \
  --precision bf16 \
  --compile \
  --wandb

  # 256/train