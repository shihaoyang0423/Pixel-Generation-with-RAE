# 像素扩散解码器：基于 RAE 的图像生成探索

**作者：施昊旸、陆修远**

复旦大学，2026 年 7 月

本项目研究如何用一个迭代式像素空间扩散模型替换表征自编码器（Representation Autoencoder，RAE）中的单步 ViT 解码器。整个系统首先使用冻结的 DiT<sup>DH</sup> 生成干净语义表征，再由独立训练的像素头合成最终的 256×256 RGB 图像。

## 项目概述

RAE 使用冻结的 DINOv2 等视觉表征编码器替代传统 VAE 编码器，在高维语义潜在空间训练扩散模型，并用一个前馈 ViT 解码器一步恢复图像。它能很好地保留结构、纹理和整体语义，但文字、人脸和精确颜色等细节仍可能出现模糊或偏差。

本项目研究的问题是：

> 能否用多步像素扩散模型替换 RAE 的一步 ViT 解码器，让语义模型与像素模型分工完成图像生成？

我们实现并系统比较了：

- RAE 语义流与像素流的同步去噪和串行解耦去噪；
- 可配置的 RAE 语义条件注入层；
- 6×768、12×1152、16×1536 三档像素头；
- 像素时间偏移（pixel time shift）与分类器自由引导（CFG）强度；
- `shared`、`split`、`traj`、`null` 四种 `s_clean` CFG；
- RAE CFG 与像素 CFG 解耦；
- AutoGuidance。

最终 16×1536 模型在 ImageNet 256×256 类条件生成上达到：

| 指标 | 结果 |
|---|---:|
| FID50k ↓ | **2.065** |
| sFID ↓ | 5.33 |
| Inception 分数 ↑ | 264.28 |
| 精确率 ↑ | 0.79 |
| 召回率 ↑ | 0.62 |

该结果证明像素扩散解码器可以作为 RAE 后端工作，但它使用约 15.32 亿总推理参数和 50+50 步 Euler 采样，仍未超过原始 RAE 的 FID 1.13。我们的主要结论是：串行解耦、模型容量和 CFG 设计都有效，但瓶颈并不只是解码器容量；DINOv2 语义表征本身缺少部分像素级信息。继续堆叠更重的像素头并不是理想的质量—效率路径。

## 方法

### 串行解耦去噪

早期版本让 RAE 语义流和像素流共享时间步并同步去噪。6×768 像素头在该设置下 FID 约为 66。两条扩散过程的状态空间不同，像素头又持续接收不稳定的中间语义状态，导致条件信号混乱。

当前方法改为三阶段串行生成：

```text
阶段一：RAE 语义空间常微分方程
z_rae ~ N(0,I) [B,768,16,16]
  → 冻结的 DiTDH-XL，50 步 Euler
  → x_rae_clean

阶段二：提取干净语义条件
x_rae_clean
  → t=0 时的 DiTDH-XL 编码器
  → s_clean [B,256,1152]

阶段三：像素空间常微分方程
z_pix ~ N(0,I) [B,3,256,256]
  → 16 层、隐藏维度 1536 的像素头，50 步 Euler
  → 图像 [B,3,256,256]
```

两组初始噪声独立采样，两条常微分方程使用独立的时间步分布和时间偏移。RAE 流先完成语义生成；像素流随后在固定 `s_clean` 条件下完成像素生成。

串行解耦将早期约 66 的 FID 降至 48–44，经过 x-prediction、结构和工程调优后，6×768 模型的无 CFG FID 进一步降至约 35。

### 流匹配

在线性路径下：

```math
x_t = (1-t)x_1 + tx_0,\qquad x_0\sim\mathcal N(0,I),\quad t\in[0,1].
```

这里 `t=0` 为干净数据端，`t=1` 为噪声端。RAE 流使用速度预测（velocity prediction）；像素流使用数据预测（data/x-prediction），再根据路径关系换算速度。相同 6×768 配置下，速度预测在第 50 个训练周期的 FID5k 为 174.84，而 x-prediction 可达到 35.69。

采样时间偏移为：

```math
\tau_\lambda(s)=\frac{\lambda s}{1+(\lambda-1)s}.
```

本文档和脚本中的 `PIX_TS` 即像素流采样阶段的 `λ`。最终模型使用 `PIX_TS=3.5`。

### 可配置条件注入

像素头由 DiT 模块构成。RAE 条件不必进入每一层，可通过 YAML 指定：

```yaml
rae_inject_layers: [6]
rae_inject_final_layer: false
use_direct_rae_injection: false
use_y_embedder: true
```

最终模型中：

- 第 6 个像素模块（从 0 开始编号）使用 `t_pix + s_clean`；
- 其余 15 个模块使用 `t_pix + y`；
- 最终输出层使用 `t_pix + y`；
- `s_clean` 来自完整 DiT<sup>DH</sup>-XL 编码器，而非直接重排 DINOv2 潜在表征。

### 可学习的 `null_s_clean`

串行框架的 CFG 不仅涉及类别标签 `y`，还涉及语义条件 `s_clean`。我们实现了四种无条件语义条件：

| 模式 | 无条件 `s_clean` | 含义 |
|---|---|---|
| `shared` | 复用有条件 `s_clean` | CFG 主要来自类别标签 |
| `split` | 对同一 RAE 潜在表征使用空类别重新提取 | 仍保留同一图像语义 |
| `traj` | 独立运行无条件 RAE 轨迹 | 两路条件来自不同图像 |
| `null` | 使用可学习的 `null_s_clean` | 真正置空语义条件 |

最终模型使用：

```yaml
class_dropout_prob: 0.1
use_s_clean_null_token: true
```

训练时，同一个丢弃掩码同步控制：

1. DiT<sup>DH</sup> 编码器的类别标签；
2. 是否将整条 `s_clean` 替换为 `null_s_clean`；
3. 像素头非注入层的类别标签。

因此，无条件分支会同时失去类别条件和 RAE 语义条件。采样时使用 `--s-clean-cfg-mode null`。

## 主要结果

### 像素头规模

| 设置 | 像素头 | 冻结的语义主干 | 引导 | FID |
|---|---:|---:|---|---:|
| 早期同步去噪 | 6×768，67M | DiT<sup>DH</sup>-S，197M | 无 | ≈66 |
| 串行基础版本 | 6×768，67M | DiT<sup>DH</sup>-S，197M | 无 | ≈35 |
| 串行中等模型 | 12×1152，295M | DiT<sup>DH</sup>-XL，839M | 无 | 5.06 |
| 12×1152 + null CFG | 12×1152，295M | DiT<sup>DH</sup>-XL，839M | 有 | 2.42 |
| **16×1536 + null CFG** | **693M** | **DiT<sup>DH</sup>-XL，839M** | **有** | **2.065** |

模型放大带来持续收益，但最佳模型总推理参数约为 839M+693M=1532M，而且需要先运行 50 步 RAE 常微分方程，再运行 50 步像素常微分方程。

### ImageNet 256×256 方法比较

下表中的其他方法指标来自各自论文。不同工作的评测协议、类别均衡采样和参数统计可能存在差异，因此该表主要用于定位量级。

| 方法 | 使用引导 | 参数量 | 训练周期 | FID ↓ | sFID ↓ | IS ↑ | 精确率 ↑ | 召回率 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DiT-XL/2 | 是 | 675M+49M | 1400 | 2.27 | 4.60 | 278.2 | 0.83 | 0.57 |
| SiT-XL/2 | 是 | 675M+49M | 1400 | 2.06 | 4.50 | 277.5 | 0.80 | 0.64 |
| REPA-XL/2 | 是 | 675M+49M | 800 | 1.42 | 4.70 | 305.7 | 0.80 | 0.64 |
| RAE-XL | 否 | 839M+415M | 800 | 1.51 | — | 242.9 | 0.79 | 0.63 |
| RAE-XL | 是 | 839M+415M | 800 | 1.13 | — | 262.6 | 0.78 | 0.67 |
| PixelGen-XL/16 | 是 | — | 160 | 1.83 | — | 293.6 | 0.79 | 0.63 |
| JiT-G/16 | 是 | 2000M | 600 | 1.82 | — | 292.6 | 0.79 | 0.62 |
| EPG-G/16 | 是 | 1391M | 1600 | 1.58 | — | 298.4 | 0.80 | 0.63 |
| PixelDiT-XL | 是 | 797M | 320 | 1.61 | 4.68 | 292.7 | 0.78 | 0.64 |
| HyperDiT-H | 是 | 952M | 600 | 1.56 | 4.73 | 306.5 | 0.80 | 0.64 |
| **像素扩散解码器（本项目）** | **是** | **839M+693M** | **50** | **2.065** | **5.33** | **264.28** | **0.79** | **0.62** |

## 消融实验

### `s_clean` CFG

6×768 模型在相同采样条件下的结果：

| 模式 | CFG | FID5k ↓ |
|---|---:|---:|
| 不使用 CFG | 1.0 | 41.980 |
| `shared` | 2.0 | 24.689 |
| `split` | 2.0 | 24.965 |
| `null` | 2.0 | **19.923** |
| `traj` | 2.0 | 37.458 |

`null` 最符合分类器自由引导的无条件定义，也取得最佳结果。

### 像素时间偏移与 CFG

16×1536 模型第 50 个训练周期、`null` 模式下的 FID5k：

| `PIX_TS` \ CFG | 1.0 | 1.2 | 1.5 | 2.0 | 2.5 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 18.958 | 14.037 | 12.103 | 12.150 | 12.575 |
| 2.0 | 12.060 | 9.354 | 9.614 | 11.096 | 12.086 |
| 3.0 | 10.570 | 8.073 | 8.742 | 10.666 | 11.856 |
| 3.2 | 10.530 | 7.991 | 8.658 | 10.617 | 11.828 |
| 3.5 | 10.559 | **7.929** | 8.573 | 10.565 | 11.788 |
| 3.7 | 10.630 | 7.924 | 8.539 | 10.532 | 11.771 |
| 4.0 | 10.789 | 7.958 | 8.513 | 10.501 | 11.752 |

`PIX_TS=3.7`、CFG=1.2 的 FID5k 略低，但差异很小。正式 FID50k 使用稳定区域内的 `PIX_TS=3.5`、CFG=1.2。

### 训练曲线

| 模型 | 周期 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12×1152 FID5k | 11.707 | 9.203 | 8.895 | 8.820 | 8.756 | 8.652 | 8.635 | 8.573 |
| 16×1536 FID5k | 11.893 | 9.062 | 8.745 | 8.774 | 8.742 | — | — | — |

### AutoGuidance

| 模型 | 较弱模型 | 基线 FID5k | 最佳 AutoGuidance FID5k |
|---|---|---:|---:|
| 6×768 | DiT<sup>DH</sup>-S 第 14 周期 | 42.740 | 36.714 |
| 12×1152 | DiT<sup>DH</sup>-S 第 14 周期 | 15.080 | 13.458 |

AutoGuidance 能提供额外修正，但不足以改变整体质量—效率结论。

## 环境

### 安装依赖

建议从 Python 3.10 和 `uv` 开始：

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# 根据服务器 CUDA 驱动选择对应安装包。
# 以下为 CUDA 12.9 示例：
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

可用以下命令安装：

```bash
conda env create -f environment.yml
conda activate rae
```

最终 16×1536 实验使用 8 张 NVIDIA H100 GPU、PyTorch 分布式数据并行和 fp32。产生最佳检查点的完整 `conda list`、CUDA、cuDNN 和驱动版本仍需从原服务器归档。因此当前两套安装说明尚未经过严格等价性验证。

## 数据与预训练权重

### 准备 ImageNet-1K

1. 从 ImageNet 官方渠道或具有合法访问权限的镜像下载 ImageNet-1K。国内环境可使用 [OpenDataLab ImageNet-1K](https://openxlab.org.cn/datasets/OpenDataLab/ImageNet-1K)。

2. 解压原始数据。

3. 参考 [REPA 数据预处理](https://github.com/sihyun-yu/REPA/tree/main/preprocessing)，将图像统一缩放并中心裁剪到 256×256：

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


正式发布前应改为命令行参数或环境变量。

### 下载 RAE 预训练模型

RAE 的解码器、DiT<sup>DH</sup> 和潜在表征归一化统计量可从 Hugging Face 下载：

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

| 资源 | 作用 | YAML 字段 |
|---|---|---|
| DINOv2-with-registers-base | 将图像编码为 `[768,16,16]` 潜在表征 | `stage_1.params.encoder_config_path` |
| DINOv2 归一化 `stat.pt` | 潜在表征归一化 | `stage_1.params.normalization_stat_path` |
| ViTXL_n08 解码器 | 训练期间可视化 RAE 潜在表征 | `stage_1.params.pretrained_decoder_path` |
| DiT<sup>DH</sup>-XL `stage2_model.pt` | 冻结的语义去噪器和编码器 | `stage_2.params.stage2_ckpt` |
| ImageNet 参考 NPZ | FID 参考统计量 | `eval.reference_npz_path` / `REF` |

### 准备像素头检查点

从头训练只加载冻结的 DiT<sup>DH</sup>-XL，像素头和 `null_s_clean` 随机初始化。复现最佳结果需要：

```text
PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/
└── checkpoints/ep-0000050.pt
```

采样器优先加载检查点中的 EMA。该检查点的公开下载地址和 SHA256 尚待补充。

### 准备 FID 参考统计量

`src/eval/fid.py` 需要参考 NPZ 至少包含 `mu` 和 `sigma`：

```bash
wget \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

## 配置

主力实验由一对 OmegaConf YAML 驱动：

```text
训练配置：
configs/stage2/training/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml

采样配置：
configs/stage2/sampling/ImageNet256/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml
```

训练和采样 YAML 的模型结构字段必须一致：

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

| 项目 | 设置 |
|---|---|
| 硬件 | 8× NVIDIA H100 |
| 精度 | fp32 |
| 最终结果检查点 | 第 50 个训练周期 |
| 全局批量大小 | 256 |
| 每卡批量大小 | 32 |
| 梯度累积 | 1 |
| 优化器 | AdamW |
| 学习率 | `2e-4` |
| Betas | `(0.9, 0.95)` |
| 权重衰减 | 0 |
| 学习率调度 | 线性；预热 50 个周期，衰减结束于第 100 个周期 |
| 最终学习率 | `2e-5` |
| EMA 衰减 | 0.9999 |
| 梯度裁剪 | 1.0 |
| 训练随机种子 | 42 |
| 像素流 | 线性路径、数据预测、速度损失 |
| 像素时间分布 | logit-normal(0.8, 0.8) |
| RAE 流 | 线性路径、速度预测 |
| 采样器 | 两条流各使用 50 步 Euler |

## 训练

主力入口：

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```


可用环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,...,7` | GPU 列表 |
| `TRAIN_CONFIG_SRC` | 1536 训练 YAML | 训练配置 |
| `SAMPLE_CONFIG_SRC` | 1536 采样 YAML | 自动 FID 配置 |
| `USE_COMPILE` | `1` | 是否启用 `torch.compile` |
| `USE_WANDB` | `1` | 配置完成后是否启用 WandB |
| `ENTITY` | 未设置 | WandB 实体 |
| `PROJECT` | 未设置 | WandB 项目 |

当前启动脚本会先执行 `unset EXPERIMENT_NAME`，因此实验名固定为：

```text
PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean
```

### 快照、续训与自动 FID5k

启动脚本会将关键 YAML、训练脚本、采样脚本和模型文件复制到 `/tmp`。训练崩溃后等待 30 秒，并从最新检查点自动恢复。

每 10 个训练周期：

1. 保存检查点；
2. 写入 `.fid_needed`；
3. 退出训练；
4. 启动脚本生成 5000 张图并计算 FID5k；
5. 写入 `fid_record.txt`；
6. 创建 `.fid_done`；
7. 自动继续训练。

自动 FID5k 使用 CFG 1.5、`PIX_TS=3.0`、`s_clean_cfg_mode=null`。它用于监控训练，并非最终最佳采样设置。

### 输出目录

```text
PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/
├── config.yaml
├── src/
├── log.txt
├── fid_record.txt
└── checkpoints/
    ├── ep-0000010.pt
    ├── ep-0000020.pt
    └── ep-0000050.pt
```

检查点包含模型、EMA、优化器、学习率调度器、训练周期和训练步数。

## 采样与评估

### CFG 与像素时间偏移扫描

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

脚本按以下关系修改临时采样 YAML：

```text
time_dist_shift_dim = round(PIX_TS² × 4096)
```

### 最佳 FID50k 单点

最终设置：

```text
检查点：            第 50 个训练周期
CFG 强度：           1.2
像素时间偏移：       3.5
s_clean 模式：       null
CFG 区间：           [0.0, 1.0]
RAE CFG 强度：       与像素 CFG 相同
样本数：             50000
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

直接调用采样器：

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

采样器输出 PNG 文件夹和包含 `arr_0` 的 NPZ。计算 FID：

```bash
cd src

python -m eval.fid \
  --arr1 /path/to/generated.npz \
  --ref /path/to/VIRTUAL_imagenet256_labeled.npz \
  --ref-mu-key mu \
  --ref-sigma-key sigma
```

计算 IS、FID、精确率和召回率：

```bash
bash cfg_all_metrics.sh
```

## 分析与结论

本项目得到两层结论。

第一，像素扩散解码器在 RAE 后端是可训练的。串行解耦稳定了条件信号；x-prediction、可配置条件注入、AdaLN-Zero、类别嵌入、可学习的 `null_s_clean` 和像素时间偏移都带来明确收益。模型从 6×768 放大到 12×1152、16×1536 时也呈现清晰规模效应。

第二，仅仅增加像素头容量不是高效路径。最佳系统比原始 RAE 更大、更慢，FID 仍更差。DINOv2 潜在表征面向语义判别学习，并不完整保存文字笔画、人脸细节和精确颜色等低层信息。像素头的任务更接近“有语义约束的重新生成”，而非从条件中恢复真实细节。CFG、时间偏移和 AutoGuidance 能改善采样，却不能增加条件本身的信息含量。

因此，该项目最重要的价值不是 FID 2.065 本身，而是提供一个完整、可测量的负面基线：

- 串行解耦是必要的；
- 可学习的空语义条件是当前 CFG 中最合理的选择；
- 模型容量和采样时间分布很关键；
- 若希望以合理代价超过 RAE 解码器，需要重新考虑语义表征与像素细节之间的信息传递，而不只是继续扩大像素解码器。

## 仓库结构

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

更细的依赖、路径、输出和清理记录见 [plans/cleanup-note.md](plans/cleanup-note.md)。
