# cleanup

本文档记录 `cleanup/repro-1536-best` 实验链路，并据此进行可复现性清理。

> 当前脚本仍包含原服务器绝对路径和少量过时注释；这些问题列在文末，尚未在本轮修改。

---

## 一、当前分支状态

当前 cleanup 分支从 `main` 创建，并依次合并：

1. `Haoyang/1536_null_s_clean`
2. `Xiuyuan/1536_sweep`

其中 `Xiuyuan/1536_sweep` 本身已经包含 `Haoyang/1536_null_s_clean` 的实验历史。当前分支保留两个显式 merge commit，用于记录整理过程。

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

---

## 二、文件依赖结构

```text
训练入口
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

```text
采样评测
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

---

## 三、训练入口脚本

文件：

```text
retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

### 3.1 默认启动方式

在原服务器目录结构下：

```bash
cd /home/shihaoyang/RAE
bash retry_train_stage3_project_free_layer_null_s_clean_1536.sh
```

### 3.2 可用环境变量

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 训练和自动 FID 使用的 GPU |
| `TRAIN_CONFIG_SRC` | 1536 training YAML 的绝对路径 | 覆盖训练配置来源 |
| `SAMPLE_CONFIG_SRC` | 1536 sampling YAML 的绝对路径 | 覆盖自动 FID 配置来源 |
| `USE_WANDB` | `1` | 设为 `0` 可关闭 WandB |
| `ENTITY` | 无 | WandB entity；与 `PROJECT` 同时存在才启用 |
| `PROJECT` | 无 | WandB project |
| `USE_COMPILE` | `1` | 是否向训练脚本传入 `--compile` |

说明：

- 脚本第 10 行先执行 `unset EXPERIMENT_NAME`，因此外部传入的 `EXPERIMENT_NAME` 当前不会生效。
- 实际实验名固定为：

  ```text
  PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean
  ```

- 数据路径、结果路径和 precision 目前不能通过环境变量覆盖，仍写死在脚本中。


---

## 四、训练配置

文件：

```text
configs/stage2/training/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean.yaml
```

### 4.1 Stage-1 RAE

```text
image [B,3,256,256]
  → DINOv2-with-registers-base
  → x1_rae [B,768,16,16]
```

配置：

```yaml
stage_1:
  target: stage1.RAE
  params:
    encoder_cls: Dinov2withNorm
    encoder_input_size: 224
    reshape_to_2d: true
    noise_tau: 0.0
```

训练循环在 `torch.no_grad()` 中调用 `rae.encode(images)`。Stage-1 不参与优化。

配置同时声明了 ViTXL decoder。它不参与 Pixel Head loss；训练时仅在 WandB 可视化 RAE latent 时可能调用 `rae.decode()`。

### 4.2 Pixel Head

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

### 4.3 DiTDH-XL backbone

```yaml
stage2_config:
  input_size: 16
  patch_size: 1
  in_channels: 768
  hidden_size: [1152, 2048]
  depth: [28, 2]
  num_heads: [16, 16]
```

预训练权重：

```text
/data/shihaoyang/pretrained/DiTs/Dinov2/wReg_base/
└── ImageNet256/DiTDH-XL/stage2_model.pt
```

`PixelDDT.__init__()` 会：

1. 实例化 `DiTwDDTHead`。
2. 从 `stage2_ckpt` 加载 DiTDH-XL 权重。
3. 将整个 `self.ddt` 设置为 `requires_grad=False`。

本实验：

```yaml
use_lora: false
```

所以 DiTDH-XL 完全冻结；实际训练对象是 Pixel Head 和 `null_s_clean`。

### 4.4 RAE 条件注入

```yaml
rae_inject_layers: [6]
rae_inject_final_layer: false
use_direct_rae_injection: false
use_y_embedder: true
```

### 4.5 同步 CFG dropout 与 `null_s_clean`

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

因此被 dropout 的样本会同时丢掉类别条件和 RAE semantic condition，形成与 CFG 推理一致的 unconditional branch。

### 4.6 双 transport 与训练 loss

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

每一步：

1. Pixel 和 RAE 独立采样时间步与噪声。
2. Pixel stream 使用真实图像 `x1_pix` 构造 noisy pixel。
3. Stage-1 编码得到干净 `x1_rae`。
4. 从干净 `x1_rae@t=0` 提取固定 `s_clean`。
5. Pixel Head 预测 pixel stream。
6. 冻结的 DiTDH 同时计算 RAE velocity prediction。
7. 总 loss：

   ```text
   mean(loss_pix) + 0.01 × mean(loss_rae)
   ```

由于本实验 `use_lora=false` 且 DiTDH 完全冻结，`loss_rae` 不会更新 DiTDH 参数；当前主要优化信号来自 `loss_pix`。

### 4.7 训练超参数

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

每卡 micro batch：

```text
global_batch_size / (GPU 数 × grad_accum_steps)
```

默认 8 GPU、无梯度累积时为每卡 32。

---

## 五、训练输出、续训与自动 FID5k

### 5.1 输出目录

实验目录：

```text
/home/shihaoyang/data/ckpts/PixDDTv2/
└── PixDDT_project_free_layer_6_16_1536_DiTDHXL_null_s_clean/
    ├── config.yaml
    ├── src/                         # 首次启动时复制的 src 工作树
    ├── log.txt                      # logger 输出，具体文件名由 create_logger 决定
    ├── fid_record.txt
    └── checkpoints/
        ├── ep-0000001.pt
        ├── ep-0000010.pt
        ├── ep-0000010.fid_needed
        ├── ep-0000010.fid_done
        └── ...
```

checkpoint 内容：

```text
step
epoch
model
ema
optimizer
scheduler
```

额外日志：

```text
/home/shihaoyang/RAE/
├── train_project_free_layer_<EXPERIMENT_NAME>.log
└── train_project_free_layer_fid_<EXPERIMENT_NAME>.log
```

### 5.2 自动续训

每次 launcher 重启时，训练脚本会扫描：

```text
<experiment_dir>/checkpoints/
```

并选择 epoch 编号最大的 `.pt` / `.ckpt` / `.safetensor` 恢复：

- model
- EMA
- optimizer
- scheduler
- epoch
- global step


### 5.3 每 10 epoch 自动 FID5k

训练脚本在每个 10 的倍数 epoch：

1. 保存 checkpoint。
2. 写入 `ep-XXXXXXX.fid_needed`。
3. 主动退出。

调用快照版 sampler：

```text
src/sample_pix_ddt_project_free_layer.py
```

固定自动评测参数：

```text
num samples:       5000
batch/GPU:         64
precision:         fp32
labels:            equal
Euler steps:       sampling YAML 中的 50
CFG scale:         1.50
CFG interval:      [0.0, 1.0]
s_clean CFG mode:  null
pixel time shift:  sampling YAML 中的 3.0
```

样本写到：

```text
${HOME}/data/samples/${EXPERIMENT_NAME}/
```

随后调用：

```bash
python -m eval.fid \
  --arr1 <generated-fid5k.npz> \
  --ref /home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz \
  --ref-mu-key mu \
  --ref-sigma-key sigma
```

结果追加到：

```text
/home/shihaoyang/data/ckpts/PixDDTv2/<EXPERIMENT_NAME>/fid_record.txt
```

最后删除 `.fid_needed` 并创建 `.fid_done`，然后继续训练。

---

## 六、独立采样配置与三阶段推理

采样配置：

```text
configs/stage2/sampling/ImageNet256/
└── PixelDDT_free_layer_6_16_1536_DiTDHXL_null_s_clean_pix_ts3.yaml
```

模型结构字段必须与 training YAML 一致：

```yaml
hidden_size: 1536
depth: 16
num_heads: 16
rae_inject_layers: [6]
rae_inject_final_layer: false
use_direct_rae_injection: false
use_y_embedder: true
use_adaln_zero: true
use_s_clean_null_token: true
stage2_config:
  hidden_size: [1152, 2048]
  depth: [28, 2]
  num_heads: [16, 16]
```

采样 YAML 的默认值：

```text
CFG scale:          1.5
CFG interval:       [0.0, 1.0]
Pixel Euler steps:  50
RAE Euler steps:    50
Pixel time shift:   sqrt(36864 / 4096) = 3.0
RAE time shift:     sqrt(196608 / 4096) ≈ 6.928
```

采样入口：

```text
src/sample_pix_ddt_project_free_layer.py
```

它从 checkpoint 中优先加载 EMA 权重，然后调用：

```text
src/train_stage3_projcet_free_layer.py
└── project_sample_ode_naive()
```

推理分三阶段：

```text
Phase 1 — RAE ODE
z_rae [B,768,16,16]
  → frozen DiTDH-XL，50-step Euler
  → x_rae_clean

Phase 2 — clean condition
x_rae_clean
  → DiTDH-XL encoder at t=0
  → s_clean [B,256,1152]

Phase 3 — Pixel ODE
z_pix [B,3,256,256]
  → 16×1536 Pixel Head，固定使用 s_clean，50-step Euler
  → RGB image [B,3,256,256]
```

当 CFG scale > 1 时，sampler 在进入三阶段流程前将 noise 和 label 扩展成 `[cond; uncond]`。

### 6.1 `s_clean` CFG 模式

CLI：

```text
--s-clean-cfg-mode {shared,split,null,traj}
```

| 模式 | unconditional `s_clean` |
|---|---|
| `shared` | 直接复用 conditional `s_clean` |
| `split` | 同一个 RAE latent，但用 null label 重新提取 |
| `null` | 使用训练得到的可学习 `null_s_clean` |
| `traj` | 运行独立 unconditional RAE trajectory，再提取条件 |

最佳 1536 单点使用：

```text
null
```

### 6.2 RAE CFG 与 Pixel CFG

默认：

```text
RAE CFG scale = Pixel CFG scale
```

可用：

```bash
--rae-cfg-scale <value>
```

进行解耦。设为 `1.0` 表示关闭 Phase-1 RAE CFG，但仍可保留 Phase-3 Pixel CFG。


### 6.3 sampler 主要参数

| CLI | 默认值 | 含义 |
|---|---:|---|
| `--config` | 必填 | sampling YAML |
| `--ckpt` | 无 | PixelDDT checkpoint；未传会使用随机 Pixel Head |
| `--sample-dir` | `samples` | 输出根目录 |
| `--per-proc-batch-size` | 32 | 每张 GPU batch |
| `--num-fid-samples` | 50000 | 最终生成数量 |
| `--num-steps` | YAML 值 | 覆盖 Euler steps |
| `--global-seed` | 0 | label pool 和 noise seed 基础值 |
| `--precision` | `fp32` | `fp32` 或 `bf16` |
| `--label-sampling` | `equal` | `equal` 或 `random` |
| `--cfg-scale` | YAML 值 | 覆盖 CFG scale |
| `--t-min` / `--t-max` | YAML 值 | CFG 生效区间 |
| `--s-clean-cfg-mode` | `shared` | `shared/split/null/traj` |
| `--rae-cfg-scale` | 与 Pixel CFG 耦合 | 单独控制 RAE CFG |
| `--naive` | flag | 兼容旧 CLI；当前 project sampler 始终使用 Euler |

### 6.4 sampler 输出

默认文件夹名：

```text
PixelDDT-project-<ckpt>-cfg<scale>-sclean-<mode>-bs<batch>-euler<steps>-<precision>/
```

其中包含：

```text
000000.png
000001.png
...
```

完成后在文件夹旁生成：

```text
<folder-name>.npz
```

NPZ 的 key 为：

```text
arr_0
```

shape 为：

```text
[num_samples, 256, 256, 3]
```

如果设置环境变量：

```bash
SAVE_FOLDER=<name>
```

则使用该名称代替 sampler 的默认文件夹名。两个 sweep 脚本都依赖这个机制确保 PNG 目录和 NPZ 文件名可预测。

---

## 七、CFG/FID5k sweep

入口：

```text
cfg_sweep.sh
```

作用：

1. 根据 `PIX_TS` 动态生成临时 sampling YAML；
2. 遍历 `CFG_SCALES`；
3. 每个点生成样本；
4. 计算 FID；
5. 将结果追加到汇总文件。

### 7.1 默认参数

```text
checkpoint epoch:  50
CFG scales:        1.0 1.2 1.5 2.0 2.5
CFG interval:      [0.0, 1.0]
s_clean mode:      null
pixel time shift:  1.0
samples/point:     5000
batch/GPU:         64
precision:         fp32
label sampling:    equal
```

基础 sampling YAML 名字包含 `pix_ts3`，其内部默认也是 3.0；但 `cfg_sweep.sh` 默认会复制并覆盖：

```yaml
misc.time_dist_shift_dim
```

使实际 sweep 的默认 pixel time shift 变成 1.0。

换算：

```text
time_dist_shift_dim = round(PIX_TS² × 4096)
```

临时配置写到：

```text
/tmp/cfg_sweep_1536_ep<EP>_pixts<PIX_TS>.yaml
```

### 7.2 可覆盖参数

| 环境变量 | 默认值 |
|---|---|
| `EXPERIMENT_NAME` | 主力 1536 实验名 |
| `EP` | `0000050` |
| `CKPT` | epoch 50 checkpoint |
| `CONFIG` | 主力 1536 sampling YAML |
| `REF` | ImageNet reference statistics |
| `PYTHON` | 原服务器 `rae` 环境 Python |
| `NPROC` | 自动读取可见 GPU 数 |
| `NUM_FID_SAMPLES` | 5000 |
| `BATCH_SIZE` | 64 |
| `PRECISION` | fp32 |
| `CFG_SCALES` | `"1.0 1.2 1.5 2.0 2.5"` |
| `T_MIN` | 0.0 |
| `T_MAX` | 1.0 |
| `S_CLEAN_CFG_MODE` | null |
| `PIX_TS` | 1 |
| `SAMPLE_BASE` | 实验 samples 下的 sweep 目录 |

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CFG_SCALES="1.0 1.1 1.2 1.3" \
PIX_TS=3.5 \
NPROC=4 \
bash cfg_sweep.sh
```

若目标 NPZ 已存在，会跳过采样并重新计算 FID。

### 7.3 输出

```text
/home/shihaoyang/data/samples/<EXPERIMENT_NAME>/
└── cfg_sweep_ep<EP>_pixts<PIX_TS>/
    ├── sclean-null-s1.20-tmin0.00-tmax1.00-pixts3.5/
    ├── sclean-null-s1.20-tmin0.00-tmax1.00-pixts3.5.npz
    └── fid_results_cfg_scale.txt
```

---

## 八、最佳单点 FID50k

入口：

```text
cfg_fid50k.sh
```

当前最佳 FID5k 点为：

```text
checkpoint:         epoch 50
CFG scale:          1.2
CFG interval:       [0.0, 1.0]
s_clean CFG mode:   null
pixel time shift:   3.5
RAE CFG scale:      未单独设置，与 Pixel CFG 耦合为 1.2
```

默认生成 50000 张图，并计算 FID50k。

### 8.1 默认运行

```bash
cd /home/shihaoyang/RAE
CUDA_VISIBLE_DEVICES=0,1,2,3 bash cfg_fid50k.sh
```

### 8.2 可覆盖参数

| 环境变量 | 默认值 |
|---|---|
| `EXPERIMENT_NAME` | 主力 1536 实验名 |
| `EP` | `0000050` |
| `CKPT` | `<experiment>/checkpoints/ep-0000050.pt` |
| `CONFIG` | 主力 1536 sampling YAML |
| `REF` | ImageNet reference statistics |
| `PYTHON` | 原服务器 `rae` 环境 Python |
| `NPROC` | 自动读取可见 GPU 数 |
| `BATCH_SIZE` | 64 |
| `NUM_FID_SAMPLES` | 50000 |
| `PRECISION` | fp32 |
| `CFG_SCALE` | 1.2 |
| `T_MIN` | 0.0 |
| `T_MAX` | 1.0 |
| `S_CLEAN_CFG_MODE` | null |
| `PIX_TS` | 3.5 |
| `RAE_CFG_SCALE` | 空；与 Pixel CFG 耦合 |
| `SAMPLE_BASE` | 根据上述参数自动生成 |

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC=4 \
CFG_SCALE=1.2 \
PIX_TS=3.5 \
S_CLEAN_CFG_MODE=null \
bash cfg_fid50k.sh
```

关闭 RAE CFG、只保留 Pixel CFG：

```bash
RAE_CFG_SCALE=1.0 bash cfg_fid50k.sh
```

### 8.3 输出

默认输出根目录：

```text
/home/shihaoyang/data/samples/<EXPERIMENT_NAME>/
└── fid50k_ep0000050_cfg1.2_sclean-null-pixts3.5/
```

其中：

```text
cfg-s1.2-tmin0.0-tmax1.0-sclean-null-pixts3.5-n50000/
cfg-s1.2-tmin0.0-tmax1.0-sclean-null-pixts3.5-n50000.npz
fid50k_result.txt
```

FID 计算依赖：

```text
src/eval/fid.py
```

---

## 九、复现所需的外部资产

```text
ImageNet-1K 256×256 train/
ImageNet-1K 256×256 val/
DINOv2-with-registers-base
DINOv2 normalization stat.pt
ViTXL_n08 decoder checkpoint（训练可视化需要）
DiTDH-XL stage2_model.pt
ImageNet FID reference statistics NPZ
训练完成的 PixelDDT ep-0000050.pt（复现最佳采样）
```
