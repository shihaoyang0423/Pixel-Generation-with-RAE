# Pixel Generation with RAE 串行去噪 — 大作业项目文档

本文档面向新加入的队友，从零讲清楚这个项目在做什么、为什么这么做，以及所有常用操作的完整步骤。

**注意！不要迷信此文档，一切从实际出发！！！**

---

## 一、背景与大框架

这个项目建立在 **RAE（Representation AutoEncoder）** 工作之上。RAE 的思路是：

1. 用预训练好的 **DINOv2** 把一张 256×256 的 RGB 图像编码成一个 16×16×768 的语义潜在特征（latent）。
2. 用一个叫 **DiT^DH**（可能在仓库某些地方被称为 DDT）的扩散模型，在这个 DINOv2 潜在空间里做去噪生成。目前主要使用DiT^DH-S，这是一个小号的非对称的编解码器 Transformer：12 层编码器（隐层宽度 384，这 12 层输出的结果就是"瓶颈层信息"）+ 2 层解码器（隐层宽度 2048）。
3. 去噪完之后，DiT^DH 产生干净的 DINOv2 latent，再送进一个 415M 参数的 ViTXL 解码器，一步直接解码出像素图像。

我们的大作业在 RAE 的基础上，用一个（也许）轻量的 **Pixel Head**（扩散网络，目前主要使用 6 层 DiT，隐层宽度 768， 67M 参数）替换掉 ViTXL 解码器，让像素的生成也通过扩散过程完成。

---

## 二、核心思路：串行解耦去噪（Sequential Decoupled Denoising）

### 早期版本的问题（同步去噪）

最早的版本（`train_stage3.py` / `PixelDDTv2.py`）中，DiT^DH 和 Pixel Head 是**同步去噪**的——两者共享同一个时间步 t，DiT^DH 每走一步就把当前的中间结果传给 Pixel Head 作为条件，两边同时前进。

问题在于：这样 Pixel Head 接收到的信息可能很杂很乱，强行共享一个 t 会让两边都受委屈。实验结果是 FID ≈ 66，效果很差。

### 我们的方案（串行去噪，即 `_project` 版本）

核心改动：**先让 DiT^DH 完整地跑完 50 步去噪，得到干净的 DINOv2 latent，再把这个干净的信息提取出来，作为固定条件喂给 Pixel Head 的 50 步去噪**。两条流水线的时间步完全解耦、独立运行。

具体分三个阶段：

```
Phase 1 — RAE ODE（50 步 Euler）
  纯噪声 z_rae_noise  →  DiT^DH 去噪  →  干净的 z_rae_clean (16×16×768)

Phase 2 — 提取条件（1 次前向，无梯度）
  z_rae_clean  →  DiT^DH 12 层编码器  →  瓶颈层特征 s_clean (16×16×384)
  （或者直接 reshape 得到原始 DINOv2 特征，见"直接注入"选项）

Phase 3 — Pixel ODE（50 步 Euler）
  纯噪声 z_pix_noise  →  Pixel Head（固定条件 s_clean）→  生成图像 (3×256×256)
```

这个改动让 FID 从同步版本的 66 下降到 **48~44**，在 Pixel Head 参数量扩大到 388M 时更可达 **35** （狗屎💩）。

---

## 三、关键的 free_layer 扩展（`_free_layer` 版本）

在串行去噪的基础上，我们进一步引入了**可配置的条件注入层**：允许通过 yaml 文件精确控制 Pixel Head 的哪些层接收来自 RAE 的语义条件，哪些层不接收，以此研究语义条件在 Pixel Head 各层的作用分布。

这是大作业的核心分析工具，也是 `_free_layer` 系列文件存在的意义。

---

## 四、文件结构与各文件说明

### 模型文件

| 文件 | 说明 |
|------|------|
| `src/stage2/models/PixelDDTv2.py` | 早期同步去噪版本，不是大作业重点 |
| `src/stage2/models/PixelDDTv2_project.py` | 串行去噪版本，每层都注入条件 |
| `src/stage2/models/PixelDDTv2_project_free_layer.py` | **当前主力**，在串行去噪基础上支持可配置的层注入 |

### 训练脚本

| 文件 | 说明 |
|------|------|
| `src/train_stage3.py` | 同步去噪的训练脚本，不是大作业重点 |
| `src/train_stage3_projcet.py` | 串行去噪版本（注意 projcet 是历史拼写错误，请保留🤣） |
| `src/train_stage3_projcet_free_layer.py` | **当前主力**，与 projcet 版逻辑完全一致，模型由 yaml 决定 |

### 采样脚本

| 文件 | 说明 |
|------|------|
| `src/sample_pix_ddt.py` | 同步去噪版本的采样 |
| `src/sample_pix_ddt_project.py` | 串行去噪版本的采样 |
| `src/sample_pix_ddt_project_free_layer.py` | **当前主力**，free_layer 版本的采样 |

### 训练配置 yaml

| 文件 | 说明 |
|------|------|
| `configs/stage2/training/PixelDDTv2_small_lora_project_free_layer.yaml` | **当前主力**，控制模型结构、训练超参、注入层配置 |

### 采样配置 yaml

| 文件 | 说明 |
|------|------|
| `configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml` | **当前主力**，采样时的配置，**必须与训练配置里的 `rae_inject_layers`、`rae_inject_final_layer` 等等等保持一致** |

### 启动脚本

| 文件 | 说明 |
|------|------|
| `retry_train_stage3_project_free_layer.sh` | **当前主力**，训练启动 + 崩溃自动重启 + 每 10 个 ckpts 进行 FID5k 自动评测 |

### 辅助工具

| 文件 | 说明 |
|------|------|
| `eval_pix_ts_sweep.py` | 批量扫描不同 pixel_time_shift 值对应的 FID5k，找到采样最优配置 |
| `record.txt` | 常用指令备忘 |

---

## 五、yaml 配置详解

**训练配置**（`configs/stage2/training/PixelDDTv2_small_lora_project_free_layer.yaml`）和**采样配置**（`configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml`）中有几个关键字段，修改时两个文件要保持一致：

```yaml
stage_2:
  target: stage2.models.PixelDDTv2_project_free_layer.PixelDDT
  params:
    depth: 6           # Pixel Head 的 DiT Block 数量（当前配置 6 层，768 维）
    hidden_size: 768   # Pixel Head 的隐层宽度
    use_lora: true     # 是否对 DiT^DH backbone 注入 LoRA
    lora_rank: 16      # LoRA rank

    # ───── 条件注入控制（free_layer 的核心配置）─────
    rae_inject_layers: [2]
    # 指定 Pixel Head 的哪几层（0-indexed）接受来自 RAE 的语义条件注入。
    # null  → 全部 depth 层都注入（等价于 _project 版本）
    # []    → 没有任何层注入
    # [0, 4, 8, 12] → 仅第 0、4、8、12 层注入（其余层只用时间步条件）

    rae_inject_final_layer: false
    # 是否对 Pixel Head 的 DDTFinalLayer（输出头）也注入 RAE 条件。
    # final_layer 是独立于 rae_inject_layers 的单独开关。

    use_direct_rae_injection: false
    # false（默认）：瓶颈层注入——干净的 DINOv2 latent 先经过 DiT^DH 的 12 层编码器提取瓶颈特征，再注入。
    # true：直接注入——干净的 DINOv2 latent 直接 reshape 后经一层线性映射注入，跳过 12 层编码器。
    direct_rae_in_channels: 768  # DINOv2 latent 的通道数，一般不需要改
```

```yaml
misc:
  time_dist_shift_dim: 4096   # pixel_time_shift = sqrt(4096/4096) = 1.0
  time_dist_shift_base: 4096   # 一般不修改这个基准值

misc_rae:
  time_dist_shift_dim: 196608  # rae_time_shift = sqrt(196608/4096) ≈ 6.93
  time_dist_shift_base: 4096
```

`pixel_time_shift` 控制 Pixel Head 去噪时时间步的疏密分布：值越大，采样时高噪声端（t≈1）步数越密集；值越小，低噪声端（t≈0）越密集。训练和采样时的 time_shift 必须保持一致，或者通过 `eval_pix_ts_sweep.py` 找到最优采样 time_shift。

（没错，这个仓库里 t=1 是噪声端， t=0 是干净端🤗）

---

## 六、如何跑训练

### 第一步：配置好 yaml

打开 `configs/stage2/training/PixelDDTv2_small_lora_project_free_layer.yaml`，根据需要修改以下字段：

- `rae_inject_layers`：决定注入哪些层
- `rae_inject_final_layer`：决定 final_layer 是否注入
- `use_direct_rae_injection`：选择瓶颈注入还是直接注入
- `use_lora`：对不对预训练好的 DiT^DH 用 LoRA
- `lora_rank`：LoRA rank，当前默认 16
- `global_batch_size`：总 batch size，**必须能被实际使用的 GPU 数量整除**。为了控制变量，不同实验间尽量保持 256 不变；若 GPU 数量导致 256 无法整除，可用相近的数如 255、258。以 ImageNet-1K（约 128 万训练样本）为例，global_batch_size=256 时大约 5000 步走完一个 Epoch。

### 第二步：修改启动脚本

打开 `retry_train_stage3_project_free_layer.sh`，**每次启动前必须检查以下几行**：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用哪几张 GPU

export EXPERIMENT_NAME="PixDDT_small_v2_project_free_layer_example"  # 实验名，ckpt 会保存在 ~/data/ckpts/{EXPERIMENT_NAME}/ 下
```

采样配置路径一般不需要改，已经指向 `_free_layer.yaml`。

### 第三步：在 tmux 里启动

强烈建议在 tmux 里跑，防止 SSH 断线后训练中断：

```bash
tmux new -s train
bash /home/shihaoyang/RAE/retry_train_stage3_project_free_layer.sh
# 按 Ctrl-B 然后按 D 可以退出 tmux 但让训练在后台继续
# 随时用 tmux attach -t train 重新进入查看进度
```

(更多指令请问 AI )

### 脚本内部做了什么

`retry_train_stage3_project_free_layer.sh` 是一个带自动重启和自动 FID 评测的循环脚本：

1. 调用 `torchrun` 启动 `src/train_stage3_projcet_free_layer.py` 进行 DDP 多卡训练。
2. 训练过程中，每到整 10 epoch（比如 ep10、ep20、ep30……），训练脚本会保存 checkpoint，然后主动写一个 `.fid_needed` 信号文件并退出。
3. 启动脚本检测到信号文件后，自动调用 `src/sample_pix_ddt_project_free_layer.py` 采样 5000 张图，计算 FID5k，结果追加写入 `~/data/ckpts/{EXPERIMENT_NAME}/fid_record.txt`，然后重新拉起训练，从上次的 checkpoint 继续。
4. 如果训练因为 OOM、NaN loss 或其他原因崩溃，脚本会等 30 秒后自动重启，从最新的 checkpoint 续训。
5. NaN loss（退出码 1）直接重启；checkpoint 文件写损坏（退出码 2）会先删掉损坏的 checkpoint 再重启。

训练日志和 FID 评测日志的文件名已通过 `EXPERIMENT_NAME` 变量自动拼接（`train_project_free_layer_${EXPERIMENT_NAME}.log` 和 `train_project_free_layer_fid_${EXPERIMENT_NAME}.log`），所以**只要在脚本顶部把 `EXPERIMENT_NAME` 改对，log 文件名会自动跟着变，不需要单独修改**。

---

## 七、如何跑采样推理（生成图像、测 FID50k）

采样推理用来在训练结束（或任意 checkpoint）后生成 50000 张图并计算 FID50k，作为正式评测指标。

### 采样命令

```bash
cd ~/RAE

export CUDA_VISIBLE_DEVICES=0,1,2,3  # 按实际可用 GPU 修改

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  src/sample_pix_ddt_project_free_layer.py \
  --config configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml \
  --ckpt /home/shihaoyang/data/ckpts/PixDDTv2/PixDDT_small_v2_project_free_layer_example/checkpoints/ep-0000050.pt \
  --sample-dir /home/shihaoyang/data/samples/PixDDT_small_v2_project_free_layer_example \
  --num-fid-samples 50000 \
  --per-proc-batch-size 64 \
  --precision fp32 \
  --label-sampling equal \
  --naive
```

参数说明：
- `--nproc_per_node=4`：用几张 GPU，与 export 了几张 GPU 保持一致。
- `--config`：采样配置 yaml，控制模型结构、采样器步数等。
- `--ckpt`：要测的 checkpoint 文件路径，按需替换。
- `--sample-dir`：生成的图像和 npz 文件保存路径，按实验名自行命名。
- `--num-fid-samples 50000`：生成 50k 张图用于计算 FID50k（快速验证可以改成 5000）。
- `--per-proc-batch-size 64`：每张 GPU 采样的 batch size，总 batch = 该值 × GPU 数量。
- `--naive`：使用 Euler 方法（50 步），与训练时 wandb 可视化一致。不加这个参数默认使用 dopri5，步数自适应。

采样完成后，`sample-dir` 下会出现：
- 一个以 checkpoint epoch 和采样参数命名的 `.npz` 文件（包含生成图像的像素数组）
- 一个同名的文件夹（包含所有生成图像的 png，供直观查看）

### 计算 FID

采样完成后，切换到 `src/` 目录，然后执行：

```bash
cd ~/RAE/src

python -m eval.fid \
  --arr1 /home/shihaoyang/data/samples/PixDDT_small_v2_project_free_layer_example/PixelDDT-project-ep-0000050-cfg1.00-bs64-euler50-fp32.npz \
  --ref /home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz \
  --ref-mu-key mu \
  --ref-sigma-key sigma
```

把 `--arr1` 后面的路径替换成你实际的 npz 文件路径。如果想指定某张 GPU 来跑（默认 0 号 GPU）：

```bash
CUDA_VISIBLE_DEVICES=3 python -m eval.fid ...
```

注意：`eval.fid` 模块在 `src/eval/fid.py` 里，必须从 `src/` 目录执行。

---

## 八、如何用 `eval_pix_ts_sweep.py` 找最优采样 time_shift

训练时用的 `pixel_time_shift`（比如训练时 ts=2.0）不一定是采样时的最优值。`eval_pix_ts_sweep.py` 可以批量扫描多个 ts 值对应的 FID5k，帮你找到最优的采样配置。

### 使用方法

打开 `eval_pix_ts_sweep.py`，修改文件顶部的用户设置区域：

```python
# 要测试的 pixel_time_shift 值列表（不需要写 dim，脚本自动换算）
PIXEL_TIME_SHIFTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # 根据情况调整粗扫列表

# 要测的 checkpoint 路径
CKPT = (
    "/home/shihaoyang/data/ckpts/PixDDTv2/你的实验名/checkpoints/ep-0000050.pt"
)

# 采样配置文件（misc.time_dist_shift_dim 会被脚本动态替换。注意，**除此之外的配置都需要与测试的模型的实际配置保持一致**）
BASE_CONFIG = (
    "configs/stage2/sampling/ImageNet256"
    "/PixDDT_small_v2_lora_project_free_layer.yaml"
)

# 输出目录（每个 ts 值的 npz、png、log 都会放在这里）
SAMPLE_ROOT = os.path.expanduser(
    "~/data/samples/你的实验名/compare_different_pix_ts"
)

NPROC_PER_NODE = 4  # GPU 数量
NUM_FID_SAMPLES = 5000  # 用 5k 快速筛选，找到最优值后再跑 50k
```

修改好之后，直接运行：

```bash
cd ~/RAE
python eval_pix_ts_sweep.py
```

脚本会依次对每个 ts 值：
1. 动态修改 yaml 里的 `time_dist_shift_dim`（`dim = ts² × 4096`）
2. 调用 `torchrun` 采样 5k 张图
3. 计算 FID5k
4. 把结果写入 `{SAMPLE_ROOT}/results.txt`

如果某个 ts 值的 npz 文件已经存在，会直接跳过采样，只算 FID（方便断点续测）。

最终 `results.txt` 大概长这样：

```
[2026-05-13 11:40:00] pix_ts1.00  dim=4096  [RESULT] FID: 50.10
[2026-05-13 12:05:00] pix_ts1.50  dim=9216  [RESULT] FID: 47.23
[2026-05-13 12:30:00] pix_ts2.00  dim=16384 [RESULT] FID: 46.33
[2026-05-13 12:55:00] pix_ts2.50  dim=25600 [RESULT] FID: 48.91
```

找到 FID 最小的那个 ts，一般就是当前 checkpoint 的最优采样 time_shift，再用它跑 50k 的采样测 FID50k。

关于 `pixel_time_shift` 的物理含义：值越大，采样时在高噪声端（t≈1）步数越密集；值越小，低噪声端（t≈0）步数越密集。最优值可能取决于训练时的噪声分布配置，两者不匹配可能会导致明显的 FID 退化。

---

## 九、采样推理的内部流程（三阶段串行）

`sample_pix_ddt_project_free_layer.py` 内部调用 `project_sample_ode_naive` 函数，整个推理过程分三个阶段顺序执行：

**Phase 1 — RAE ODE 去噪（50 步 Euler）**

从随机高斯噪声出发，用 DiT^DH（含 LoRA，如果训练时开了的话）跑 50 步 Euler 积分，得到干净的 DINOv2 latent `z_rae_clean`，形状 `[B, 768, 16, 16]`。

**Phase 2 — 提取条件特征（一次前向，不参与 ODE）**

把 `z_rae_clean` 传入 `model.extract_clean_s()`：
- 如果 `use_direct_rae_injection=false`（默认）：`z_rae_clean` 经过 DiT^DH 的 12 层编码器，输出瓶颈特征 `s_clean`，形状 `[B, 256, 384]`。这 256 个 token 对应 16×16 的空间位置，每个 token 是该位置经过全局注意力融合后的语义表示。
- 如果 `use_direct_rae_injection=true`：`z_rae_clean` 直接 reshape 成 `[B, 256, 768]` 后经一层 768→384 的线性映射，得到 `[B, 256, 384]`，跳过 12 层编码器。

**Phase 3 — Pixel ODE 去噪（50 步 Euler）**

从另一组随机高斯噪声出发，用 Pixel Head 跑 50 步 Euler 积分。Pixel Head 的每一步中，在 `rae_inject_layers` 指定的层里会把 `s_clean` 与当前时间步嵌入混合，作为 AdaLN 条件注入该层；不在列表里的层只使用时间步条件。`final_layer` 是否注入由 `rae_inject_final_layer` 单独控制。最终输出 3×256×256 的 RGB 生成图像。

两套噪声（RAE 用的和 Pixel 用的）完全独立采样，这是"解耦"的体现。

---

## 十、训练过程监控

训练过程可以通过以下方式监控：

### WandB

训练脚本默认开启 WandB（`--wandb` 参数），每隔 10 步记录一次 loss。每 2000 步会在 WandB 上上传一组 3×3 共 9 张生成图像（用 Euler 50 步，训练中的 wandb 采样和最终采样推理完全一致）。

队友需要用自己的 WandB 账号接管监控，步骤如下（仅供参考）：

1. 在 [https://wandb.ai](https://wandb.ai) 注册账号，进入账号设置页面，找到并复制 API Key。

2. 在服务器上登录：

```bash
wandb login
# 粘贴 API Key 并回车
```

3. 在启动训练前 export 以下环境变量（也可以直接写进 `retry_train_stage3_project_free_layer.sh` 顶部）：

```bash
export WANDB_API_KEY=你的API_Key
export WANDB_ENTITY=你的用户名或团队名   # WandB 上的 entity，对应项目的归属方
export WANDB_PROJECT=你想用的项目名      # 比如 pixel_ddt，first_project
```

配置好之后正常运行 `bash retry_train_stage3_project_free_layer.sh` 即可，WandB 会自动把 loss 曲线和生成图像同步到你自己的账号下。

### 训练日志

```bash
tail -f ~/RAE/train_project_free_layer_example.log
```

FID 中间评测日志：

```bash
tail -f ~/RAE/train_project_free_layer_fid_example.log
```

历次 FID5k 评测结果汇总：

```bash
cat ~/data/ckpts/{EXPERIMENT_NAME}/fid_record.txt
```

---

## 十一、清理 checkpoint（节省磁盘）

长期训练会产生大量 checkpoint 文件，以下命令在 checkpoint 目录里，保留 20 的倍数 epoch、25 的倍数 epoch 以及最新的一个，其余全部删除：

```bash
cd ~/data/ckpts/{EXPERIMENT_NAME}/checkpoints

# 第一步：预览哪些文件会被删（不实际删除）
m=0; for f in ep-[0-9]*.pt; do n=$((10#${f:3:7})); (( n>m )) && m=$n; done
for f in ep-[0-9]*.pt; do n=$((10#${f:3:7})); (( n%20==0 || n%25==0 || n==m )) || echo "将删除: $f"; done

# 第二步：确认无误后正式删除
for f in ep-[0-9]*.pt; do n=$((10#${f:3:7})); (( n%20==0 || n%25==0 || n==m )) || rm -v "$f"; done
```

---

## 十二、参数量速查

| 模块 | 参数量 | 状态 |
|------|-------:|------|
| DiT^DH backbone（DiT^DH-S） | ~197M | 冻结（frozen） |
| LoRA 增量（rank=16） | ~0.9M | 可训练 |
| Pixel Head（6层，768维） | ~67M | 可训练 |
| Pixel Head（16层，1152维，大参数版） | ~388M | 可训练 |
| Stage-1 RAE decoder（ViTXL_n08，仅解码） | 415M | 冻结，推理时不需要；训练时仅用于 WandB 可视化（将 RAE 流生成的 latent 解码为图像用于对比展示） |
| DINOv2 encoder（训练时提特征用） | 86M | 冻结，采样时不需要 |

推理时实际运行的模型（默认 6 层 768 维配置）：DiT^DH（197M）+ Pixel Head（67M）= **约 264M**，不包含 ViTXL 解码器（Pixel Head 采样时不需要它）。

---

## 十三、常见问题

**Q：为什么脚本名字里是 `projcet` 而不是 `project`？**

历史拼写错误，已沿用成习惯，不要改，改了会和已有 checkpoint 里保存的代码路径对不上。

**Q：FID 计算报 `No module named 'eval'`？**

必须从 `src/` 目录执行，或者在命令前加 `PYTHONPATH=/home/shihaoyang/RAE/src`。

**Q：训练崩溃了但日志显示 `rc=1` 而不是我期望的退出码？**

这是 `torchrun` 的正常行为：任意一个 worker 以非零码退出，`torchrun` 会 SIGTERM 其他 worker 然后自己以 code 1 退出，原始退出码被屏蔽。FID 评测的触发用文件信号（`.fid_needed`）而不是退出码，所以不受影响。

**Q：`pixel_time_shift` 怎么和 yaml 里的 `time_dist_shift_dim` 对应？**

换算公式：`time_shift = sqrt(dim / base)`，其中 base 固定为 4096。反过来：`dim = time_shift² × 4096`。例如 ts=2.0 对应 dim=16384，ts=0.25 对应 dim=256。

**Q：`use_direct_rae_injection` 和 `rae_inject_layers` 可以同时设置吗？**

可以，两者是独立的。`use_direct_rae_injection` 控制条件特征的**来源**（是否跳过 12 层编码器），`rae_inject_layers` 控制条件特征**注入 Pixel Head 的哪几层**。

**Q：训练和采样的 `rae_inject_layers`、`rae_inject_final_layer`、`use_direct_rae_injection` 不一致会怎样？**

模型加载时会按照 yaml 里的参数初始化，如果这些配置和 checkpoint 里保存的模型结构不一致（例如新增了 `s_raw_projector` 但 checkpoint 里没有），会有部分参数随机初始化，生成效果很差，或者干脆报错跑不起来。务必保持训练配置和采样配置里相关字段一致。

---

## 十四、从零搭环境

### Conda 环境配置

新服务器上可能没有预装 Conda，需要先自行下载安装 Miniconda 或 Anaconda ，具体安装方法问 AI 。

安装好 Conda 之后，按照仓库根目录下的 `README.md`（即 RAE 官方 README）的 **Environment → Dependency Setup** 部分创建并配置 `rae` 虚拟环境：

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# 安装 PyTorch（根据服务器的 CUDA 版本选择对应的安装命令，详见 README.md）
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# 安装其余依赖
uv pip install -r requirements.txt
```

具体的 PyTorch 版本和 CUDA 版本对应关系以 `README.md` 中的说明为准，不同服务器的 CUDA 版本可能不同，安装命令需要对应调整。

---

### 数据集：ImageNet

训练需要 ImageNet-1K 的训练集，数据量约 140GB。国内推荐使用浦源平台镜像下载，速度快很多：

> [https://openxlab.org.cn/datasets/OpenDataLab/ImageNet-1K](https://openxlab.org.cn/datasets/OpenDataLab/ImageNet-1K)

下载完成后需要做以下处理：

1. **解压**原始压缩包（tar 格式）。

2. **预处理**：参考 [REPA 仓库的 preprocessing 指南](https://github.com/sihyun-yu/REPA/tree/main/preprocessing)，把图像 resize + center crop 到 256×256。核心命令如下：

```bash
# 把原始 ImageNet 转成 256×256 的 ZIP 格式
python dataset_tools.py convert \
    --source=[你的ImageNet解压路径]/ILSVRC/Data/CLS-LOC/train \
    --dest=[目标路径]/images \
    --resolution=256x256 \
    --transform=center-crop-dhariwal
```

3. **整理文件夹结构**：REPA 预处理完成后，还需要把训练集整理成按数字编号（`0000`～`0999`）命名的 1000 个类别子文件夹，每个文件夹内是 `img????????.png` 格式的图片，例如：

```
/data/data/imagenet/256/train/
├── 0000/
│   ├── img00000000.png
│   └── ...
├── 0001/
├── ...
└── 0999/（共 1000 个类别文件夹）
```

如果是新服务器需要自己准备数据，预处理完成后让 AI 写一个简单的脚本，把图像整理成上述结构。对应路径在训练配置 yaml 和启动脚本里的 `--data-path` 参数中指定（**所有训练集路径需与真实路径保持一致**）。

---

### 预训练模型

训练需要以下几个预训练模型，均需从 HuggingFace 下载。国内网络访问 HuggingFace 较慢，建议使用镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 使用 HuggingFace 镜像
```

然后用 `huggingface-cli` 下载（需要先 `pip install huggingface_hub`）：

```bash
huggingface-cli download <repo_id> --local-dir <本地目录>
```

具体需要下载的模型和路径见 `RAEDME.md` （RAE repo 的 README 文档）的 `Download Pre-trained Models` 部分。

---

## 十五、Git 协作流程

### 分支规范

```
main                        ← 稳定公共代码，只接受合并，不直接在这里改
├── 名字/实验描述         ← 实验分支，例如 Haoyang/direct-inject
└── 名字/实验描述         ← 实验分支，例如 Xiuyuan/new-arch
```

每人在自己的分支上独立工作，互不干扰。想把某个实验的改动共享给对方，才合并到 `main`。

---

### 开始一个新实验（每次都这样做）

```bash
# 进入仓库目录（你的路径可能是 ~/Pixel-Generation-with-RAE ，建议将 Pixel-Generation-with-RAE/ 文件名改为 RAE/）
cd ~/RAE   # 或者 cd ~/Pixel-Generation-with-RAE

# 先把 main 更新到最新
git checkout main
git pull origin main

# 从最新的 main 开一个自己的实验分支
git checkout -b 你的名字/实验描述   # 例如：git checkout -b Xiuyuan/direct_injection
```

---

### 日常保存进度

```bash
# 查看改了哪些文件
git status

# 把所有改动加进暂存区（或只加某个文件：git add 文件路径）
git add .

# 写一条 commit，描述你做了什么
git commit -m "简短描述，例如：try direct injection with 2-layer projector"

# 推送到远端（第一次推新分支需要 -u，之后直接 git push 即可）
git push -u origin 你的分支名
```

> **什么时候 commit？** 不需要等实验跑完再 commit。改了代码就 commit 一次，这样万一改坏了可以随时回退。commit 要频繁、粒度小。

---

### 拿到队友的最新代码

```bash
# 更新远端信息
git fetch origin

# 方式一：把队友的分支直接拉下来看（只读，不改动自己的分支）
git checkout origin/队友的分支名

# 方式二：把 main 上的最新公共代码合并到你当前的实验分支
git checkout 你的分支名
git merge main
# 如果有冲突，git 会提示哪些文件冲突，打开文件找到 <<<< ==== >>>> 标记，
# 手动选择保留哪一方的代码，改完后：
git add .
git commit -m "merge main into 你的分支名"
```

---

### 把自己的实验成果共享到 main（让队友也能用）

```bash
# 先确保自己的分支是最新的
git checkout 你的分支名
git push origin 你的分支名

# 切回 main，拉最新
git checkout main
git pull origin main

# 合并自己的分支
git merge 你的分支名

# 推送到远端
git push origin main
```

合并前两人最好口头确认一下，避免同时往 main 合并造成冲突。

---

### 万一改坏了，回退代码

```bash
# 查看最近的提交记录，找到你想回退到的那一条的 commit hash（前7位即可）
git log --oneline -10

# 方式一：软回退（代码改动保留，只是撤销 commit）
git reset --soft HEAD~1        # 撤销最近1次 commit
git reset --soft 某个commit的hash

# 方式二：硬回退（彻底恢复到那个版本，之后的改动全丢）
git reset --hard 某个commit的hash
```

---

### 不会被提交的东西

以下内容已经在 `.gitignore` 里忽略，不会被提交：

| 类型 | 举例 |
|------|------|
| 训练日志 | `*.log` |
| 模型权重 | `*.pth`、`checkpoints/` |
| 数据集 | `data/`、`/data/` |
| 采样结果 | `ckpts/` |
| IDE 缓存 | `.cursor/` |
| 论文 PDF | `*.pdf` |

如果你发现 `git status` 里出现了上面这些文件，说明 `.gitignore` 没有生效，**不要 `git add .`**，先检查原因再操作。
