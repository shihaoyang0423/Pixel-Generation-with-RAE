export CUDA_VISIBLE_DEVICES=3
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/sample_pix.py \
  --config /home/zhangchi/RAE/configs/stage2/training/PixelDDT.yaml \
  --sample-dir samples \
  --precision fp32 \
  --label-sampling equal \
  --ckpt /data/zhangchi/ckpts/PixDDT/768dim/checkpoints/ep-0000039.pt \
  --naive 