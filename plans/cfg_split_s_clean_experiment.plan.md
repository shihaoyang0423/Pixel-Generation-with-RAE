---
name: CFG Split s_clean Experiment
overview: 规范化 split_s_clean 采样实验：比较 Pixel CFG 中 cond/uncond 两路共用 class-conditioned s_clean 与分别使用 class/null-conditioned s_clean 的 FID 差异，优先作为当前 SOTA checkpoint 上的 sampling-only ablation。
todos:
  - id: merge-main
    content: 将最新 main 合并到 Xiuyuan/CFG_split_s_clean，确保实验基于最新 free_layer 代码
    status: completed
  - id: document-plan
    content: 在 plans/ 中记录 split_s_clean 的动机、数据流、实现点、sweep 矩阵和判读规则
    status: completed
  - id: sampler-toggle
    content: 给 project_sample_ode_naive 增加 s_clean_cfg_mode，支持 shared/split 两种 Phase 2 CFG conditioning
    status: completed
  - id: cli-toggle
    content: 给 sample_pix_ddt_project_free_layer.py 增加 --s-clean-cfg-mode，并将模式写入输出 tag
    status: completed
  - id: sweep-script
    content: 新增 cfg_split_s_clean_sweep.sh，固定当前 SOTA checkpoint，成对跑 shared 与 split 的 FID5k
    status: completed
  - id: run-ablation
    content: 运行第一轮 shared/split cfg scale sweep，记录 FID、最佳 scale 和是否需要细扫
    status: pending
isProject: false
---

# CFG Split s_clean 实验计划

## 背景

当前 project/free_layer 采样是三阶段串行流程：

```text
Phase 1: z_rae -> RAE ODE -> x_rae_clean
Phase 2: x_rae_clean -> extract_clean_s(...) -> s_clean
Phase 3: z_pix -> Pixel ODE conditioned on s_clean -> image
```

CFG 采样时，`z_pix`、`z_rae`、`y` 都会扩成 `[cond; uncond]`：

```text
z_pix = [z_pix, z_pix]
z_rae = [z_rae, z_rae]
y     = [y_class, y_null]
```

Pixel CFG 的最终差分为：

```text
guided = uncond + cfg_scale * (cond - uncond)
```

因此，`cond - uncond` 是否干净，直接决定 CFG scale sweep 是否有意义。

## 核心问题

在原先 shared s_clean 逻辑中，Phase 2 只用 conditional half 提取一次 `s_clean`，再复制给 cond/uncond 两路：

```text
s_clean_cond = extract_clean_s(x_rae_clean_cond, y_class)
s_clean      = [s_clean_cond, s_clean_cond]

Pixel cond:   y_class, s_clean_cond
Pixel uncond: y_null,  s_clean_cond
```

这个 uncond path 不是完全 unconditional，因为它虽然用了 `y_null`，但仍然拿到了 class-conditioned `s_clean_cond`。如果 Pixel Head 的注入层强依赖 `s_clean`，则 CFG 差分可能被冲淡。

split_s_clean 的想法是：

```text
s_clean_cond = extract_clean_s(x_rae_clean_cond, y_class)
s_clean_null = extract_clean_s(x_rae_clean_cond, y_null)
s_clean      = [s_clean_cond, s_clean_null]

Pixel cond:   y_class, s_clean_cond
Pixel uncond: y_null,  s_clean_null
```

这样被 RAE feature 注入的 Pixel 层也参与 `cond - uncond`，理论上会让 guidance direction 更接近真正的 conditional/unconditional 差分。

## 当前合并后的代码状态

本分支已合并最新 `main`。合并后，`src/train_stage3_projcet_free_layer.py` 的 `project_sample_ode_naive()` 中 Phase 2 已经是 hard-coded split 行为：

```python
x_rae_half = x_rae_clean[:B]
y_cond = y[:B]
y_uncond = y[B:]
s_clean_cond = model.extract_clean_s(x_rae_half, y_cond)
s_clean_uncond = model.extract_clean_s(x_rae_half, y_uncond)
s_clean = torch.cat([s_clean_cond, s_clean_uncond], dim=0)
```

这说明最新主线已经包含了 split_s_clean 的初步实现，但还缺少可复现实验所需的对照开关和 sweep 记录：

- 不能从 CLI 选择 `shared` 或 `split`
- 输出目录/tag 不区分 s_clean mode
- 没有专用 sweep 脚本成对比较两种模式
- shared baseline 需要能在同一份代码里复现

因此，本实验分支后续应把 hard-coded split 改造成显式实验开关，而不是继续只保留单一行为。

## 实现计划

### 1. `src/train_stage3_projcet_free_layer.py`

给 `project_sample_ode_naive()` 增加参数：

```python
s_clean_cfg_mode="shared"
```

建议支持两个值：

| mode | 行为 | 用途 |
|---|---|---|
| `shared` | `s_clean = [s_clean_cond, s_clean_cond]` | 复现旧 baseline |
| `split` | `s_clean = [s_clean_cond, s_clean_null]` | 本实验候选方案 |

推荐默认值为 `shared`，保证旧命令和旧结果可复现。实验脚本显式传 `split`。

如果当前 main 已经 hard-coded split，需要把 Phase 2 改成：

```python
if using_cfg:
    B = B_full // 2
    x_rae_half = x_rae_clean[:B]
    y_cond = y[:B]

    s_clean_cond = model.extract_clean_s(x_rae_half, y_cond)

    if s_clean_cfg_mode == "shared":
        s_clean = torch.cat([s_clean_cond, s_clean_cond], dim=0)
    elif s_clean_cfg_mode == "split":
        y_uncond = y[B:]
        s_clean_uncond = model.extract_clean_s(x_rae_half, y_uncond)
        s_clean = torch.cat([s_clean_cond, s_clean_uncond], dim=0)
    else:
        raise ValueError(...)
else:
    s_clean = model.extract_clean_s(x_rae_clean, y)
```

注意：`split` 模式下仍然建议使用 `x_rae_clean[:B]` 来提取 cond/null 两份 `s_clean`，不要改用 `x_rae_clean[B:]`。Phase 1 CFG 后两半本应是相同 guided RAE latent；本实验只隔离测试 `extract_clean_s` 的 label 条件差异。

### 2. `src/sample_pix_ddt_project_free_layer.py`

新增 CLI：

```bash
--s-clean-cfg-mode shared|split
```

并传入：

```python
project_sample_ode_naive(
    ...,
    s_clean_cfg_mode=args.s_clean_cfg_mode,
)
```

输出目录名需要包含 mode，避免 shared/split 的 `.npz` 撞名：

```text
PixelDDT-project-ep-XXXX-cfg2.00-sclean-shared-bs64-euler50-fp32
PixelDDT-project-ep-XXXX-cfg2.00-sclean-split-bs64-euler50-fp32
```

如果 sweep 脚本使用 `SAVE_FOLDER`，则由 sweep 脚本负责把 mode 放进 tag。

### 3. `cfg_split_s_clean_sweep.sh`

新建专用脚本，不复用现有 `cfg_sweep.sh`，避免污染旧实验。

建议配置区：

```bash
SOURCE_EXPERIMENT=PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDH-XL
CKPT_ROOT=/home/shihaoyang/data/ckpts/PixDDTv2/${SOURCE_EXPERIMENT}
CKPT=${CKPT_ROOT}/checkpoints/ep-0000050.pt
CONFIG=${CKPT_ROOT}/config.yaml
SAMPLE_BASE=/home/shihaoyang/data/samples/${SOURCE_EXPERIMENT}/cfg_split_s_clean_fid5k_ep0000050
REF=/path/to/VIRTUAL_imagenet256_labeled.npz
NPROC=4
NUM_FID_SAMPLES=5000
BATCH_SIZE=64
PRECISION=fp32
```

注意：这个 SOTA checkpoint 是 `hidden_size=1152`、`depth=12`、`stage2_ckpt=DiTDH-XL` 的实验。仓库默认的 `configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml` 目前不是这个结构，不能直接拿来加载该 ckpt。为了保证复现，split_s_clean sweep 使用 checkpoint 目录保存的 `config.yaml`。

核心函数：

```bash
run_combo MODE SCALE T_MIN T_MAX
```

tag 命名：

```text
sclean-{mode}-s{scale}-tmin{t_min}-tmax{t_max}
```

结果文件：

```text
fid_results_split_s_clean.txt
```

记录字段：

```text
mode  scale  t_min  t_max  FID5k  npz_path
```

## 第一轮实验矩阵

如果之前 shared baseline 已知最佳约为 `cfg=2.5, t_min=0.3, t_max=1.0`，第一轮优先固定 `t_min=0.3, t_max=1.0`：

```text
shared:
  cfg = 2.5

split:
  cfg = 1.5
  cfg = 1.75
  cfg = 2.0
  cfg = 2.25
  cfg = 2.5
```

如果需要更稳健的第一轮：

```text
shared:
  cfg = 2.0, 2.5, 3.0

split:
  cfg = 1.25, 1.5, 1.75, 2.0, 2.5
```

split 后 guidance direction 可能更强，所以最佳 scale 可能低于 shared。

## 判读规则

### split 低 scale 优于 shared best

说明旧 shared s_clean 确实让 uncond path 泄漏了 class-conditioned feature，CFG 差分被冲淡。下一步应围绕 split 的最佳 scale 做细扫。

### split 高 scale 变差，但低 scale 变好

这是最正常、也最有信息量的结果。说明 split 增强了 CFG direction，原来的 scale 过大，需要下调。

### split 全面变差

说明 `s_clean_null` 对当前 Pixel Head 是训练时没见过或不稳定的 conditioning 分布。此时 sampling-only split 不适合直接作为 SOTA 采样方案，但它仍然提示后续可以做 paired fine-tune：

```text
cond path:   y_class, s_clean_cond
uncond path: y_null,  s_clean_null
```

让模型训练时见过这种成对结构。

## 与后续实验的关系

split_s_clean 是 sampling-only ablation，优先依赖当前 SOTA checkpoint。它不需要重新训练，因此适合作为 CFG 优化的第一步。

如果 split 有收益，再考虑：

- 与 RAE/Pix 分离 CFG scale 组合
- 在 split 最佳 scale 附近细扫 `t_min`
- 做 split_s_clean paired fine-tune

如果 split 没收益，则不要继续围绕它投入训练，优先转向 RAE CFG off/weak 与 Pixel CFG 分离实验。
