# Pixel Diffusion Decoder with Representation Autoencoders<br><sub>Sequential RAE-to-Pixel Generation in PyTorch</sub>

### [Project Report](#project-report) | [RAE Paper](https://arxiv.org/abs/2510.11690) | [RAE Project Page](https://rae-dit.github.io/) | [Original RAE Repository](https://github.com/bytetriper/RAE)

This repository explores replacing the one-step ViT decoder in a Representation Autoencoder (RAE) with an iterative pixel-space diffusion model. The resulting system first generates a clean semantic representation with a frozen DiT<sup>DH</sup>, then uses a separately trained Pixel Head to synthesize the final 256×256 RGB image.

> [**Diffusion Transformers with Representation Autoencoders**](https://arxiv.org/abs/2510.11690)<br>
> [Boyang Zheng](https://bytetriper.github.io/), [Nanye Ma](https://willisma.github.io/), [Shengbang Tong](https://tsb0601.github.io/), [Saining Xie](https://www.sainingxie.com)<br>
> New York University

本项目由施昊旸、陆修远完成，项目报告为《像素扩散解码器：基于 RAE 的图像生成探索》（复旦大学，2026 年 7 月）。

## Overview

RAE 使用冻结的 DINOv2 等视觉表征 encoder 替代传统 VAE encoder，在高维语义 latent 空间训练扩散模型，并用一个前馈 ViT decoder 一步恢复图像。它能很好地保留结构、纹理和整体语义，但文字、人脸和精确颜色等细节仍可能出现模糊或偏差。

本项目研究一个直接的问题：

> 能否用多步像素扩散模型替换 RAE 的一步 ViT decoder，让语义模型与像素模型分工完成图像生成？

我们实现并系统比较了：

- RAE/Pixel 同步去噪与串行解耦去噪；
- 可配置的 RAE condition 注入层；
- 6×768、12×1152、16×1536 三档 Pixel Head；
- Pixel time shift 与 CFG scale；
- `shared/split/traj/null` 四种 `s_clean` CFG；
- RAE/Pixel CFG 解耦；
- AutoGuidance。

最终 16×1536 模型在 ImageNet 256×256 类条件生成上达到：

| Metric | Result |
|---|---:|
| FID50k ↓ | **2.065** |
| sFID ↓ | 5.33 |
| Inception Score ↑ | 264.28 |
| Precision ↑ | 0.79 |
| Recall ↑ | 0.62 |

该结果证明像素扩散解码器可以作为 RAE 后端工作，但它使用约 1.53B 总推理参数和 50+50 步 Euler 采样，仍未超过原始 RAE 的 FID 1.13。我们的主要结论是：串行解耦、模型容量和 CFG 设计都有效，但瓶颈并不只是 decoder 容量；DINOv2 语义表征本身缺少部分像素级信息。继续堆叠更重的 Pixel Head 并不是理想的质量—效率路径。

## Method

### Sequential Decoupled Denoising

早期版本让 RAE 语义流和 Pixel Head 像素流共享时间步并同步去噪。6×768 Pixel Head 在该设置下 FID 约为 66。两条扩散过程的状态空间不同，Pixel Head 又持续接收不稳定的中间语义状态，导致条件信号混乱。

当前方法改为三阶段串行生成：

```text
Phase 1 — RAE semantic ODE
z_rae ~ N(0,I) [B,768,16,16]
  → frozen DiTDH-XL, 50-step Euler
  → x_rae_clean

Phase 2 — Clean condition extraction
x_rae_clean
  → DiTDH-XL encoder at t=0
  → s_clean [B,256,1152]

Phase 3 — Pixel ODE
z_pix ~ N(0,I) [B,3,256,256]
  → 16-layer, 1536-hidden Pixel Head, 50-step Euler
  → image [B,3,256,256]
```

两组初始噪声独立采样，两条 ODE 使用独立 timestep 分布和 time shift。RAE 流先完成语义生成；Pixel 流随后在固定 `s_clean` 条件下完成像素生成。

串行解耦将早期约 66 的 FID 降至 48–44，经过 x-prediction、结构和工程调优后，6×768 模型的无 CFG FID 进一步降至约 35。

### Flow Matching

在线性路径下：

```math
x_t = (1-t)x_1 + tx_0,\qquad x_0\sim\mathcal N(0,I),\quad t\in[0,1].
```

这里 `t=0` 为干净数据端，`t=1` 为噪声端。RAE 流使用 velocity prediction；Pixel 流使用 data/x-prediction，再根据路径关系换算速度。相同 6×768 配置下，v-prediction 在 epoch 50 的 FID5k 为 174.84，而 x-prediction 可达到 35.69。

采样 time shift 为：

```math
\tau_\lambda(s)=\frac{\lambda s}{1+(\lambda-1)s}.
```

README 和脚本中的 `PIX_TS` 即 Pixel 流采样阶段的 `λ`。最终模型使用 `PIX_TS=3.5`。

### Free-layer Condition Injection

Pixel Head 使用 DiT blocks。RAE condition 不必进入每一层，可通过 YAML 指定：

```yaml
rae_inject_layers: [6]
rae_inject_final_layer: false
use_direct_rae_injection: false
use_y_embedder: true
```

最终模型中：

- Pixel block 6（0-indexed）使用 `t_pix + s_clean`；
- 其余 15 个 blocks 使用 `t_pix + y`；
- final layer 使用 `t_pix + y`；
- `s_clean` 来自完整 DiT<sup>DH</sup>-XL encoder，而非直接 reshape DINOv2 latent。

### Learned `null_s_clean` CFG

串行框架的 CFG 不仅涉及类别标签 `y`，还涉及语义条件 `s_clean`。我们实现了四种 unconditional condition：

| Mode | Unconditional `s_clean` | Interpretation |
|---|---|---|
| `shared` | 复用 conditional `s_clean` | CFG 主要来自类别标签 |
| `split` | 同一 RAE latent，用 null label 重新提取 | 仍保留同一图像语义 |
| `traj` | 独立 unconditional RAE trajectory | 两路 condition 来自不同图像 |
| `null` | 可学习的 `null_s_clean` | 真正置空语义条件 |

最终模型使用：

```yaml
class_dropout_prob: 0.1
use_s_clean_null_token: true
```

训练时，同一个 dropout mask 同步控制：

1. DiT<sup>DH</sup> encoder 的 class label；
2. 是否将整条 `s_clean` 替换为 `null_s_clean`；
3. Pixel Head 非注入层的 class label。

因此 unconditional branch 同时失去类别与 RAE semantic condition。采样时使用 `--s-clean-cfg-mode null`。

## Main Results

### Scaling the Pixel Head

| Setting | Pixel Head | Frozen backbone | Guidance | FID |
|---|---:|---:|---|---:|
| Early synchronous | 6×768, 67M | DiT<sup>DH</sup>-S, 197M | No | ≈66 |
| Sequential baseline | 6×768, 67M | DiT<sup>DH</sup>-S, 197M | No | ≈35 |
| Sequential medium | 12×1152, 295M | DiT<sup>DH</sup>-XL, 839M | No | 5.06 |
| 12×1152 + null CFG | 12×1152, 295M | DiT<sup>DH</sup>-XL, 839M | Yes | 2.42 |
| **16×1536 + null CFG** | **693M** | **DiT<sup>DH</sup>-XL, 839M** | **Yes** | **2.065** |

模型放大带来持续收益，但最佳模型总推理参数约为 839M+693M=1.532B，而且需要先运行 50 步 RAE ODE，再运行 50 步 Pixel ODE。

### Comparison on ImageNet 256×256

下表中的其他方法指标来自各自论文，评测协议、class-balanced sampling 和参数统计可能存在差异，因此主要用于定位量级。

| Method | Guidance | Parameters | Epochs | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DiT-XL/2 | Yes | 675M+49M | 1400 | 2.27 | 4.60 | 278.2 | 0.83 | 0.57 |
| SiT-XL/2 | Yes | 675M+49M | 1400 | 2.06 | 4.50 | 277.5 | 0.80 | 0.64 |
| REPA-XL/2 | Yes | 675M+49M | 800 | 1.42 | 4.70 | 305.7 | 0.80 | 0.64 |
| RAE-XL | No | 839M+415M | 800 | 1.51 | — | 242.9 | 0.79 | 0.63 |
| RAE-XL | Yes | 839M+415M | 800 | 1.13 | — | 262.6 | 0.78 | 0.67 |
| PixelGen-XL/16 | Yes | — | 160 | 1.83 | — | 293.6 | 0.79 | 0.63 |
| JiT-G/16 | Yes | 2B | 600 | 1.82 | — | 292.6 | 0.79 | 0.62 |
| EPG-G/16 | Yes | 1391M | 1600 | 1.58 | — | 298.4 | 0.80 | 0.63 |
| PixelDiT-XL | Yes | 797M | 320 | 1.61 | 4.68 | 292.7 | 0.78 | 0.64 |
| HyperDiT-H | Yes | 952M | 600 | 1.56 | 4.73 | 306.5 | 0.80 | 0.64 |
| **Pixel Diffusion Decoder (ours)** | **Yes** | **839M+693M** | **50** | **2.065** | **5.33** | **264.28** | **0.79** | **0.62** |

## Ablations

### `s_clean` CFG

6×768 模型、相同采样条件下：

| Mode | CFG | FID5k ↓ |
|---|---:|---:|
| No CFG | 1.0 | 41.980 |
| `shared` | 2.0 | 24.689 |
| `split` | 2.0 | 24.965 |
| `null` | 2.0 | **19.923** |
| `traj` | 2.0 | 37.458 |

`null` 最符合 classifier-free guidance 的无条件定义，也取得最佳结果。

### Pixel Time Shift × CFG

16×1536、epoch 50、`null` mode 的 FID5k：

| `PIX_TS` \ CFG | 1.0 | 1.2 | 1.5 | 2.0 | 2.5 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 18.958 | 14.037 | 12.103 | 12.150 | 12.575 |
| 2.0 | 12.060 | 9.354 | 9.614 | 11.096 | 12.086 |
| 3.0 | 10.570 | 8.073 | 8.742 | 10.666 | 11.856 |
| 3.2 | 10.530 | 7.991 | 8.658 | 10.617 | 11.828 |
| 3.5 | 10.559 | **7.929** | 8.573 | 10.565 | 11.788 |
| 3.7 | 10.630 | 7.924 | 8.539 | 10.532 | 11.771 |
| 4.0 | 10.789 | 7.958 | 8.513 | 10.501 | 11.752 |

`PIX_TS=3.7, CFG=1.2` 的 FID5k 略低，但差异很小。正式 FID50k 使用稳定区域内的 `PIX_TS=3.5, CFG=1.2`。

### Training Curves

| Model | Epoch 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12×1152 FID5k | 11.707 | 9.203 | 8.895 | 8.820 | 8.756 | 8.652 | 8.635 | 8.573 |
| 16×1536 FID5k | 11.893 | 9.062 | 8.745 | 8.774 | 8.742 | — | — | — |

### AutoGuidance

| Model | Bad model | Baseline FID5k | Best AG FID5k |
|---|---|---:|---:|
| 6×768 | DiT<sup>DH</sup>-S epoch 14 | 42.740 | 36.714 |
| 12×1152 | DiT<sup>DH</sup>-S epoch 14 | 15.080 | 13.458 |

AutoGuidance 能提供额外修正，但不足以改变整体质量—效率结论。

## Environment

### Dependency Setup

推荐参考上游 RAE，从 Python 3.10 和 `uv` 开始：

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# 根据服务器 CUDA 驱动选择对应 wheel。
# 以下为上游 RAE README 的 CUDA 12.9 示例：
uv pip install torch==2.8.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu129

uv pip install -r requirements.txt
```

仓库中的 `environment.yml` 记录了另一套较早环境：

```text
Python 3.10
PyTorch 2.6.0
torchvision 0.21.0
CUDA 12.1
```

可用：

```bash
conda env create -f environment.yml
conda activate rae
```

最终 16×1536 实验使用 8 张 NVIDIA H100 GPU、PyTorch DDP 和 fp32。产生最佳 checkpoint 的完整 `conda list`、CUDA/cuDNN 和驱动版本仍需从原服务器归档。因此当前两套安装说明尚未经过严格等价性验证。

## Data & Model Preparation

### Prepare ImageNet-1K

1. 从 ImageNet 官方渠道或具有合法访问权限的镜像下载 ImageNet-1K。国内环境可使用 [OpenDataLab ImageNet-1K](https://openxlab.org.cn/datasets/OpenDataLab/ImageNet-1K)。

2. 解压原始数据。

3. 参考 [REPA preprocessing](https://github.com/sihyun-yu/REPA/tree/main/preprocessing)，将图像统一 resize + center crop 到 256×256：

   ```bash
   python dataset_tools.py convert \
     --source=/path/to/ILSVRC/Data/CLS-LOC/train \
     --dest=/path/to/imagenet/256/train \
     --resolution=256x256 \
     --transform=center-crop-dhariwal
   ```

4. 整理为 `torchvision.datasets.ImageFolder` 可读取的类别目录：

   ```text
   /path/to/imagenet/256/
   ├── train/
   │   ├── 0000/
   │   │   ├── img00000000.png
   │   │   └── ...
   │   ├── 0001/
   │   └── ...
   └── val/
       ├── 0000/
       ├── 0001/
       └── ...
   ```

当前 launcher 仍硬编码：

```text
/data/shihaoyang/data/imagenet/256/train
```

正式发布前应改为 CLI 或环境变量。

### Download RAE Pre-trained Models

上游 RAE 在 Hugging Face 发布 RAE decoders、DiT<sup>DH</sup> 和 latent normalization statistics：

```bash
pip install huggingface_hub

hf download nyu-visionx/RAE-collections \
  --local-dir models
```

下载单个文件：

```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models
```

可选 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

主力训练需要：

| Asset | Purpose | YAML field |
|---|---|---|
| DINOv2-with-registers-base | Image → `[768,16,16]` latent | `stage_1.params.encoder_config_path` |
| DINOv2 normalization `stat.pt` | Latent normalization | `stage_1.params.normalization_stat_path` |
| ViTXL_n08 decoder | RAE visualization during training | `stage_1.params.pretrained_decoder_path` |
| DiT<sup>DH</sup>-XL `stage2_model.pt` | Frozen semantic denoiser/encoder | `stage_2.params.stage2_ckpt` |
| ImageNet reference NPZ | FID reference moments | `eval.reference_npz_path` / `REF` |

当前 YAML 中的原服务器布局：

```text
/data/shihaoyang/pretrained/
├── encoders/dinov2-with-registers-base/
├── decoders/dinov2/wReg_base/ViTXL_n08/model.pt
├── stats/dinov2/wReg_base/imagenet1k/stat.pt
└── DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt
```

### Prepare the PixelDDT Checkpoint

从头训练只加载冻结的 DiT<sup>DH</sup>-XL，Pixel Head 和 `null_s_clean` 随机初始化。复现最佳结果需要：

```text
PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/
└── checkpoints/ep-0000050.pt
```

sampler 优先加载 checkpoint 中的 EMA。该 checkpoint 的公开下载地址和 SHA256 尚待补充。

### Prepare FID Reference Statistics

`src/eval/fid.py` 需要 reference NPZ 至少包含 `mu` 和 `sigma`：

```bash
wget \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

## Config-based Initialization

主力实验由一对 OmegaConf YAML 驱动：

```text
Training:
configs/stage2/training/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml

Sampling:
configs/stage2/sampling/ImageNet256/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml
```

训练与采样 YAML 的模型结构字段必须一致：

```yaml
stage_2:
  target: stage2.models.PixelDDTv2_project_free_layer.PixelDDT
  params:
    hidden_size: 1536
    depth: 16
    num_heads: 16
    use_lora: false
    rae_inject_layers: [6]
    rae_inject_final_layer: false
    use_direct_rae_injection: false
    use_y_embedder: true
    use_adaln_zero: true
    use_s_clean_null_token: true
```

主要训练配置：

| Item | Value |
|---|---|
| Hardware | 8× NVIDIA H100 |
| Precision | fp32 |
| Epoch used for final result | 50 |
| Global batch size | 256 |
| Batch/GPU | 32 |
| Gradient accumulation | 1 |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Betas | `(0.9, 0.95)` |
| Weight decay | 0 |
| LR schedule | Linear, warmup 50 epochs, decay end 100 |
| Final LR | `2e-5` |
| EMA decay | 0.9999 |
| Gradient clipping | 1.0 |
| Training seed | 42 |
| Pixel transport | Linear, data prediction, velocity loss |
| Pixel time distribution | logit-normal(0.8, 0.8) |
| RAE transport | Linear, velocity prediction |
| Sampler | 50-step Euler for each stream |

## Training

主力入口：

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

原服务器目录结构下：

```bash
cd /home/shihaoyang/RAE

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

可用环境变量：

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,...,7` | GPU list |
| `TRAIN_CONFIG_SRC` | 1536 training YAML | Training config |
| `SAMPLE_CONFIG_SRC` | 1536 sampling YAML | Automatic FID config |
| `USE_COMPILE` | `1` | Enable `torch.compile` |
| `USE_WANDB` | `1` | Enable WandB when configured |
| `ENTITY` | unset | WandB entity |
| `PROJECT` | unset | WandB project |

当前 launcher 会先 `unset EXPERIMENT_NAME`，因此实验名固定为：

```text
PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean
```

### Snapshot, Resume, and Automatic FID5k

launcher 启动时将关键 YAML、训练脚本、采样脚本和模型文件复制到 `/tmp`。训练崩溃后等待 30 秒，并从最新 checkpoint 自动恢复。

每 10 epochs：

1. 保存 checkpoint；
2. 写入 `.fid_needed`；
3. 退出训练；
4. launcher 生成 5000 张图并计算 FID5k；
5. 写入 `fid_record.txt`；
6. 创建 `.fid_done`；
7. 自动继续训练。

自动 FID5k 使用 CFG 1.5、`PIX_TS=3.0`、`s_clean_cfg_mode=null`。它用于监控训练，并非最终最佳采样设置。

### Outputs

```text
/home/shihaoyang/data/ckpts/PixDDTv2/
└── PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/
    ├── config.yaml
    ├── src/
    ├── log.txt
    ├── fid_record.txt
    └── checkpoints/
        ├── ep-0000010.pt
        ├── ep-0000020.pt
        └── ep-0000050.pt
```

checkpoint 包含 model、EMA、optimizer、scheduler、epoch 和 step。

## Sampling and Evaluation

### CFG / Pixel Time Shift Sweep

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC=4 \
CFG_SCALES="1.0 1.2 1.5 2.0 2.5" \
PIX_TS=3.5 \
bash cfg_sweep.sh
```

常用覆盖参数：

```text
CKPT, CONFIG, REF, PYTHON, NPROC
NUM_FID_SAMPLES, BATCH_SIZE, PRECISION
CFG_SCALES, T_MIN, T_MAX
S_CLEAN_CFG_MODE, PIX_TS, SAMPLE_BASE
```

脚本按以下关系修改临时 sampling YAML：

```text
time_dist_shift_dim = round(PIX_TS² × 4096)
```

### Best FID50k Point

最终设置：

```text
checkpoint:         epoch 50
CFG scale:          1.2
pixel time shift:   3.5
s_clean mode:       null
CFG interval:       [0.0, 1.0]
RAE CFG scale:      coupled to Pixel CFG
samples:            50000
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC=4 \
CFG_SCALE=1.2 \
PIX_TS=3.5 \
S_CLEAN_CFG_MODE=null \
bash cfg_fid50k.sh
```

直接调用 sampler：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  src/sample_pix_ddt_project_free_layer.py \
  --config configs/stage2/sampling/ImageNet256/PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml \
  --ckpt /path/to/ep-0000050.pt \
  --sample-dir /path/to/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 64 \
  --precision fp32 \
  --label-sampling equal \
  --cfg-scale 1.2 \
  --t-min 0.0 \
  --t-max 1.0 \
  --s-clean-cfg-mode null \
  --naive
```

sampler 输出 PNG 文件夹和包含 `arr_0` 的 NPZ。FID：

```bash
cd src

python -m eval.fid \
  --arr1 /path/to/generated.npz \
  --ref /path/to/VIRTUAL_imagenet256_labeled.npz \
  --ref-mu-key mu \
  --ref-sigma-key sigma
```

完整 IS/FID/Precision/Recall 入口：

```bash
bash cfg_all_metrics.sh
```

## Discussion

本项目得到两层结论。

第一，像素扩散解码器在 RAE 后端是可训练的。串行解耦稳定了条件信号；x-prediction、free-layer injection、AdaLN-Zero、类别嵌入、learned `null_s_clean` 和 pixel time shift 都带来明确收益。模型从 6×768 放大到 12×1152、16×1536 时也呈现清晰规模效应。

第二，仅仅增加 Pixel Head 容量不是高效路径。最佳系统比原始 RAE 更大、更慢，FID 仍更差。DINOv2 latent 是面向语义判别学习的表征，并不完整保存文字笔画、人脸细节和精确颜色等低层信息。Pixel Head 的任务更接近“有语义约束的重新生成”，而非从 condition 中恢复真实细节。CFG、time shift 和 AutoGuidance 能改善采样，却不能增加 condition 本身的信息含量。

因此，该项目最重要的价值不是 FID 2.065 本身，而是提供一个完整、可测量的负面基线：

- 串行解耦是必要的；
- learned null condition 是当前 CFG 中最合理的选择；
- 模型容量和采样时间分布很关键；
- 若希望以合理代价超过 RAE decoder，需要重新考虑语义表征与像素细节之间的信息传递，而不只是继续扩大像素 decoder。

## Repository Structure

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
cfg_sweep.sh
cfg_fid50k.sh
cfg_all_metrics.sh

configs/stage2/training/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml

configs/stage2/sampling/ImageNet256/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml

src/
├── train_stage3_projcet_free_layer.py
├── sample_pix_ddt_project_free_layer.py
├── stage1/
├── stage2/
│   ├── models/
│   │   ├── DDT.py
│   │   └── PixelDDTv2_project_free_layer.py
│   └── transport/
└── eval/
    ├── fid.py
    └── all_metrics_torch_fidelity.py
```

更细的依赖、路径、输出和 cleanup 审计见 [plans/cleanup-note.md](plans/cleanup-note.md)。

## Current Reproducibility Gaps

正式发布前仍需：

1. 移除 `/home/shihaoyang`、`/data/shihaoyang` 硬编码；
2. 归档最佳实验的完整 Conda/CUDA/cuDNN/GPU 信息；
3. 发布 epoch-50 checkpoint、下载地址和 SHA256；
4. 提交原始 FID50k/all-metrics 输出；
5. 记录 ImageNet preprocessing 与 reference NPZ hash；
6. 修正 launcher 中残留的 12×1152 注释；
7. 让 data/model/output roots 真正可配置；
8. 增加模型实例化、strict checkpoint load、少量采样和 FID smoke tests。

## Project Report

本 README 根据项目报告《像素扩散解码器：基于 RAE 的图像生成探索》整理。报告包含更完整的相关工作、数学推导、实验对比、消融和讨论。

报告作者：

```text
施昊旸
陆修远
复旦大学
2026-07-01
```

## Citation

使用本仓库时，请引用 RAE 原论文，并保留对原始代码仓库 [bytetriper/RAE](https://github.com/bytetriper/RAE) 的说明：

```bibtex
@article{zheng2025representationautoencoders,
  title={Diffusion Transformers with Representation Autoencoders},
  author={Boyang Zheng and Nanye Ma and Shengbang Tong and Saining Xie},
  journal={arXiv preprint arXiv:2510.11690},
  year={2025}
}
```

## Acknowledgement

This repository is built upon the official [bytetriper/RAE](https://github.com/bytetriper/RAE) codebase for *Diffusion Transformers with Representation Autoencoders*. We thank the RAE authors for releasing their PyTorch implementation, pretrained RAE components, DiT<sup>DH</sup> models, and training/evaluation infrastructure.

The upstream RAE implementation is itself built upon:

- [SiT](https://github.com/willisma/sit) — diffusion implementation and training codebase.
- [DDT](https://github.com/MCG-NJU/DDT) — parts of the DiT<sup>DH</sup> implementation.
- [LightningDiT](https://github.com/hustvl/LightningDiT/) — PyTorch DiT components.
- [MAE](https://github.com/facebookresearch/mae) — ViT decoder architecture.

Please consult the [original RAE repository](https://github.com/bytetriper/RAE) for the upstream implementation, pretrained models, project updates, and authoritative RAE documentation.

## License

This project follows the repository's [MIT License](LICENSE).
