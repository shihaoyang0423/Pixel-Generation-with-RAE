torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config /home/zhangchi/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
  --sample-dir samples \
  --precision fp32 \
  --label-sampling equal