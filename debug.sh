export CUDA_VISIBLE_DEVICES=1
export EXPERIMENT_NAME=x0_pred_ds_corr
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/train_stage3.py \
  --config /home/zhangchi/RAE/configs/stage2/training/x0_pred_768.yaml \
  --data-path /data/zhangchi/data/imagenet/256/train \
  --results-dir /data/zhangchi/ckpts/PixDDT \
  --precision bf16 \
  --compile \
  --wandb

  # 256/train