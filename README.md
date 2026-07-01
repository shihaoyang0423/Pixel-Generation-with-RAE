# Pixel Generation with RAE

本仓库基于 [Diffusion Transformers with Representation Autoencoders (RAE)](https://github.com/bytetriper/RAE) 的官方 PyTorch 实现，研究一种串行的图像生成流程：先在 RAE/DINOv2 表征空间完成去噪，再用独立的 Pixel Head 在像素空间生成 256×256 图像。

当前可复现主线是：

```text
DiTDH-XL RAE stream
  → clean DINOv2 latent
  → clean semantic condition
  → 16-layer / 1536-hidden Pixel Head
  → RGB image
```

上游 RAE 使用冻结的 DINOv2 等表征 encoder 与训练得到的 ViT decoder 构成 Representation Autoencoder，并在表征 latent 空间训练 DiT。本项目保留其 Stage-1 RAE 和 DiT<sup>DH</sup> 实现，在此基础上增加 Pixel Head、串行双流训练、free-layer condition injection 和 learned `null_s_clean` CFG。

## 当前最佳实验

| 组件 | 配置 |
|---|---|
| 图像分辨率 | 256×256 |
| Stage-1 encoder | DINOv2-with-registers-base |
| RAE denoiser | 冻结的 DiTDH-XL |
| DiTDH-XL | 28 层 encoder（1152）+ 2 层 decoder（2048） |
| Pixel Head | 16 层、hidden size 1536、16 heads |
| RAE 条件注入 | 只注入 Pixel block 6（0-indexed） |
| Pixel final layer | 不注入 RAE condition |
| 非注入层条件 | timestep + class label |
| LoRA | 关闭 |
| CFG unconditional condition | 可学习的 `null_s_clean` |
| 正式采样点 | epoch 50、CFG 1.2、pixel time shift 3.5 |
| CFG 区间 | `[0.0, 1.0]` |
| `s_clean` CFG mode | `null` |

配置：

```text
Pixel Head:       16 blocks × hidden size 1536
RAE backbone:     DiTDH-XL（28 encoder blocks + 2 decoder blocks）
RAE injection:    仅注入 Pixel block 6（0-indexed）
final layer:      不注入 RAE condition
class condition:  非注入层使用 t + y
initialization:   AdaLN-Zero
LoRA:             关闭，DiTDH-XL 完全冻结
CFG null branch:  可学习的 null_s_clean
```

训练入口：

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

采样实验入口：

```text
cfg_sweep.sh      # CFG scale / pixel time shift 的 FID5k 扫描
cfg_fid50k.sh     # 最佳单点的正式 FID50k
```


## 文件依赖结构

### 训练入口

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
├── training YAML
│   └── configs/stage2/training/
│       └── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml
├── training Python
│   └── src/train_stage3_projcet_free_layer.py
├── model
│   └── src/stage2/models/PixelDDTv2_project_free_layer.py
├── Stage-1 RAE encoder
│   ├── src/stage1/rae.py
│   └── src/stage1/encoders/dinov2.py
├── DiTDH implementation
│   └── src/stage2/models/DDT.py
├── transport / ODE
│   └── src/stage2/transport/
├── distributed, optimizer, resume, WandB helpers
│   └── src/utils/
└── 每 10 epoch 自动 FID5k
    ├── sampling YAML
    │   └── configs/stage2/sampling/ImageNet256/
    │       └── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml
    ├── src/sample_pix_ddt_project_free_layer.py
    └── src/eval/fid.py
```

### 采样评测

```text
cfg_sweep.sh 或 cfg_fid50k.sh
├── sampling YAML（同上）
├── src/sample_pix_ddt_project_free_layer.py
│   ├── src/train_stage3_projcet_free_layer.py
│   │   └── project_sample_ode_naive()
│   ├── src/stage2/models/PixelDDTv2_project_free_layer.py
│   ├── src/stage2/models/DDT.py
│   ├── src/stage2/transport/
│   └── src/utils/
└── src/eval/fid.py
```

## 训练

训练入口脚本：

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

### 默认启动方式

在原服务器目录结构下：

```bash
cd /home/shihaoyang/RAE
bash retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

### 可用环境变量

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 训练和自动 FID 使用的 GPU |
| `TRAIN_CONFIG_SRC` | 1536 training YAML 的绝对路径 | 覆盖训练配置来源 |
| `SAMPLE_CONFIG_SRC` | 1536 sampling YAML 的绝对路径 | 覆盖自动 FID 配置来源 |
| `USE_WANDB` | `1` | 设为 `0` 可关闭 WandB |
| `ENTITY` | 无 | WandB entity；与 `PROJECT` 同时存在才启用 |
| `PROJECT` | 无 | WandB project |
| `USE_COMPILE` | `1` | 是否向训练脚本传入 `--compile` |


### 训练配置

```text
configs/stage2/training/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml
```

#### Pixel Head

```yaml
stage_2:
  target: stage2.models.PixelDDTv2_project_free_layer.PixelDDT
  params:
    input_size: 256
    patch_size: 16
    in_channels: 3
    hidden_size: 1536
    depth: 16
    num_heads: 16
    mlp_ratio: 4.0
```

说明：

- 256×256 RGB 输入以 16×16 patch 切分，得到 16×16=256 个 token。
- Pixel Head 共 16 个 Transformer block。
- 每个 token hidden size 为 1536。
- 使用 SwiGLU、RoPE、RMSNorm 和 AdaLN-Zero。

#### RAE 条件注入

```yaml
rae_inject_layers: [6]
rae_inject_final_layer: false
use_direct_rae_injection: false
use_y_embedder: true
```

#### 同步 CFG dropout 与 `null_s_clean`

```yaml
class_dropout_prob: 0.1
use_s_clean_null_token: true
```

模型维护一个可学习参数：

```text
null_s_clean [1,256,1152]
```

训练时每个 batch 只采样一次 `force_drop_ids`，并同步控制：

1. DiTDH encoder 提取 `s_clean` 时的 class label；
2. 是否将整条 `s_clean` 替换成可学习 `null_s_clean`；
3. Pixel Head 非注入层使用的 class label。

#### 双 transport 与训练 loss

Pixel stream：

```yaml
transport:
  params:
    path_type: Linear
    prediction: data
    loss_weight: velocity
    time_dist_type: logit-normal_0.8_0.8
```

RAE stream：

```yaml
transport_rae:
  params:
    path_type: Linear
    prediction: velocity
    loss_weight: null
    time_dist_type: logit-normal_0_1
```

训练函数：

```text
src/train_stage3_projcet_free_layer.py
└── project_training_losses()
```


#### 训练超参数

```yaml
training:
  epochs: 1400
  global_batch_size: 256
  grad_accum_steps: 1
  ema_decay: 0.9999
  num_workers: 16
  log_interval: 10
  checkpoint_interval: 1
  sample_every: 2000
  clip_grad: 1.0
  global_seed: 42
  rae_loss_weight: 0.01
  optimizer:
    lr: 2.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
  scheduler:
    type: linear
    warmup_epochs: 50
    decay_end_epoch: 100
    base_lr: 2.0e-4
    final_lr: 2.0e-5
    warmup_from_zero: false
```


## 环境

### 依赖配置

推荐参考上游 RAE 的 GPU 环境，从 Python 3.10 和 `uv` 开始安装：

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

uv pip install torch==2.8.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu129

uv pip install -r requirements.txt
```


## 数据与预训练权重

### 数据集准备

1. 下载 ImageNet-1K。原实验文档使用过 [OpenDataLab ImageNet-1K 镜像](https://openxlab.org.cn/datasets/OpenDataLab/ImageNet-1K)；也可以使用具有合法访问权限的其他来源。

2. 解压原始数据。

3. 将图像预处理为 256×256。可参考 [REPA preprocessing](https://github.com/sihyun-yu/REPA/tree/main/preprocessing)，使用仓库根目录的 `dataset_tools.py` 做 Dhariwal center crop：

   ```bash
   python dataset_tools.py convert \
     --source=/path/to/ILSVRC/Data/CLS-LOC/train \
     --dest=/path/to/imagenet/256/train \
     --resolution=256x256 \
     --transform=center-crop-dhariwal
   ```

4. 整理为 `torchvision.datasets.ImageFolder` 可读取的类别目录。当前实验约定类别目录使用 `0000` 至 `0999`：

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

5. 将训练入口的 `--data-path` 指向 `train/`。当前主力 launcher 仍硬编码：

   ```text
   /data/shihaoyang/data/imagenet/256/train
   ```

训练阶段还会读取 YAML 中的 `eval.data_path` 以构造 validation `ImageFolder`。当前配置对应：

```text
/data/shihaoyang/data/imagenet/256/val
```

训练 transform：

```text
center crop to 256
random horizontal flip
ToTensor → [0,1]
```

### 下载预训练 RAE

上游 RAE 在 Hugging Face 发布了 RAE decoder、DiT<sup>DH</sup> 和 latent normalization statistics。下载完整集合：

```bash
pip install huggingface_hub

hf download nyu-visionx/RAE-collections \
  --local-dir models
```

只下载单个文件：

```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models
```

在 Hugging Face 访问较慢的网络环境中，可按需配置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新运行 `hf download` 或 `huggingface-cli download`。

主力训练需要以下外部资产：

| 资产 | 作用 | 当前 YAML 字段 |
|---|---|---|
| DINOv2-with-registers-base | 将 ImageNet 图像编码为 `[768,16,16]` latent | `stage_1.params.encoder_config_path` / `encoder_params.dinov2_path` |
| DINOv2 normalization `stat.pt` | latent normalization | `stage_1.params.normalization_stat_path` |
| ViTXL_n08 decoder | 训练期间解码 RAE latent 做可视化 | `stage_1.params.pretrained_decoder_path` |
| DiTDH-XL `stage2_model.pt` | 冻结的 RAE denoiser 与 condition encoder | `stage_2.params.stage2_ckpt` |
| ImageNet reference NPZ | FID reference moments | `eval.reference_npz_path` / sweep 的 `REF` |

主力 YAML 当前记录的原服务器布局为：

```text
/data/shihaoyang/pretrained/
├── encoders/dinov2-with-registers-base/
├── decoders/dinov2/wReg_base/ViTXL_n08/model.pt
├── stats/dinov2/wReg_base/imagenet1k/stat.pt
└── DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt
```

### FID Reference Statistics

当前 `src/eval/fid.py` 需要 reference NPZ 至少包含：

```text
mu
sigma
```

ImageNet-256 ADM reference 下载地址为：

```bash
wget \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

下载后将 `REF` 或 YAML 中的 `reference_npz_path` 指向该文件，并在正式结果记录中保存它的 SHA256。

## 训练

### 训练配置

主力配置：

```text
configs/stage2/training/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml
```

主要优化参数：

```yaml
training:
  epochs: 1400
  global_batch_size: 256
  grad_accum_steps: 1
  ema_decay: 0.9999
  checkpoint_interval: 1
  sample_every: 2000
  clip_grad: 1.0
  global_seed: 42
  rae_loss_weight: 0.01
  optimizer:
    lr: 2.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
  scheduler:
    type: linear
    warmup_epochs: 50
    decay_end_epoch: 100
    base_lr: 2.0e-4
    final_lr: 2.0e-5
```

Pixel stream 使用 data prediction；RAE stream 使用 velocity prediction：

```yaml
transport:
  params:
    path_type: Linear
    prediction: data
    loss_weight: velocity
    time_dist_type: logit-normal_0.8_0.8

transport_rae:
  params:
    path_type: Linear
    prediction: velocity
    loss_weight: null
    time_dist_type: logit-normal_0_1
```

总 loss：

```text
mean(loss_pix) + 0.01 × mean(loss_rae)
```

本实验中 DiTDH-XL 完全冻结，因此主要可训练对象是 Pixel Head 和 `null_s_clean`。

### 启动训练

原服务器目录结构下：

```bash
cd /home/shihaoyang/RAE

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

可用环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,...,7` | GPU 列表 |
| `TRAIN_CONFIG_SRC` | 主力 training YAML 绝对路径 | 覆盖训练配置 |
| `SAMPLE_CONFIG_SRC` | 主力 sampling YAML 绝对路径 | 覆盖自动 FID 配置 |
| `USE_COMPILE` | `1` | 是否启用 `torch.compile` |
| `USE_WANDB` | `1` | 是否尝试启用 WandB |
| `ENTITY` | 无 | WandB entity |
| `PROJECT` | 无 | WandB project |

WandB 只有在 `USE_WANDB=1` 且 `ENTITY`、`PROJECT` 均存在时启用。

当前 launcher 会先 `unset EXPERIMENT_NAME`，所以实验名实际固定为：

```text
PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean
```

这属于待清理行为。

### 快照和自动重启

launcher 启动时会将以下文件复制到 `/tmp`：

```text
training YAML
sampling YAML
train_stage3_projcet_free_layer.py
sample_pix_ddt_project_free_layer.py
PixelDDTv2_project_free_layer.py
retry_snap_bootstrap.py
```

训练崩溃后会等待 30 秒，并从最新 checkpoint 自动恢复。快照只冻结上述关键文件；其他模块和 Python/CUDA 环境仍来自 live repo。

### 输出

```text
/home/shihaoyang/data/ckpts/PixDDTv2/
└── PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/
    ├── config.yaml
    ├── src/
    ├── log.txt
    ├── fid_record.txt
    └── checkpoints/
        ├── ep-0000001.pt
        ├── ep-0000010.pt
        └── ...
```

checkpoint 包含：

```text
model
ema
optimizer
scheduler
epoch
step
```

## 训练期间自动 FID5k

每到 10 的倍数 epoch，训练脚本会：

1. 保存 checkpoint；
2. 创建 `.fid_needed`；
3. 主动退出；
4. launcher 采样 5000 张图并计算 FID；
5. 写入 `fid_record.txt`；
6. 创建 `.fid_done`；
7. 自动继续训练。

自动 FID5k 使用：

```text
CFG scale:          1.5
pixel time shift:   3.0
s_clean CFG mode:   null
CFG interval:       [0.0, 1.0]
Euler steps:        50 + 50
precision:          fp32
label sampling:     equal
```

它用于监控训练，不是最终最佳采样点。


## 最佳单点 FID50k

入口：

```bash
bash cfg_fid50k.sh
```

当前模型最佳：

```text
epoch:              50
CFG scale:          1.2
pixel time shift:   3.5
s_clean mode:       null
CFG interval:       [0.0, 1.0]
RAE CFG scale:      与 Pixel CFG 耦合
samples:            50000
```

如需关闭 RAE CFG、只保留 Pixel CFG：

```bash
RAE_CFG_SCALE=1.0 bash cfg_fid50k.sh
```

主要覆盖变量：

```text
EP
CKPT
CONFIG
REF
PYTHON
NPROC
BATCH_SIZE
NUM_FID_SAMPLES
PRECISION
CFG_SCALE
T_MIN
T_MAX
S_CLEAN_CFG_MODE
PIX_TS
RAE_CFG_SCALE
SAMPLE_BASE
```

输出包括：

```text
50000 PNG images
generated samples .npz（key: arr_0）
fid50k_result.txt
```

FID 由以下模块计算：

```bash
python -m eval.fid \
  --arr1 <generated.npz> \
  --ref <imagenet-reference-stats.npz> \
  --ref-mu-key mu \
  --ref-sigma-key sigma
```


## `s_clean` 模式

| 模式 | unconditional `s_clean` 来源 |
|---|---|
| `shared` | 复用 conditional `s_clean` |
| `split` | 同一 RAE latent，用 null label 重新提取 |
| `null` | 可学习的 `null_s_clean` |
| `traj` | 独立 unconditional RAE trajectory |

最佳 1536 checkpoint 使用 `null`。该模式要求训练配置包含：

```yaml
use_s_clean_null_token: true
```

## 实验历史

本次 cleanup 只保留当前支持的 16×1536 入口。早期同步去噪、1152 Pixel Head、CFG dropout、CFG split 和 AutoGuidance 等实验仍保存在相应 Git 分支和提交历史中，不再全部堆放在主分支根目录。

主要实验分支：

```text
Xiuyuan/CFG_dropout
Xiuyuan/CFG_scale_sweep
Xiuyuan/rae-pixel-cfg-split
Xiuyuan/CFG_split_s_clean
Xiuyuan/1152_null_s_clean
Haoyang/1536_null_s_clean
Xiuyuan/1536_sweep
Xiuyuan/auto_guidance
```

## 参考文献

本项目建立在 RAE 代码和模型之上。使用本仓库时，请同时引用 RAE 原论文，并在代码来源中保留对 [bytetriper/RAE](https://github.com/bytetriper/RAE) 的说明：

```bibtex
@article{zheng2025representationautoencoders,
  title={Diffusion Transformers with Representation Autoencoders},
  author={Boyang Zheng and Nanye Ma and Shengbang Tong and Saining Xie},
  journal={arXiv preprint arXiv:2510.11690},
  year={2025}
}
```

本仓库当前仍处于课程项目/研究代码 cleanup 阶段。正式实验指标和本项目自身的引用信息将在复现验证完成后补充。

