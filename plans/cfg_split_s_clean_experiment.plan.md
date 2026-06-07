---
name: CFG s_clean (split / null) Experiment
overview: Pixel CFG 中 uncond 半边的 s_clean 该怎么取的实验。三种模式 shared / split / null 已全部实现并接到训练内评测、独立采样、sweep 脚本。当前主线是在 6x768 小模型上从头训练一个带可学习 null s_clean token 的版本（use_s_clean_null_token=true），用 --s-clean-cfg-mode null 让 Pixel Head 能做真正的 CFG，并和无 null token 的 [2] baseline 对照。
todos:
  - id: merge-main
    content: 将最新 main 合并到 Xiuyuan/CFG_split_s_clean
    status: completed
  - id: sampler-toggle
    content: project_sample_ode_naive 增加 s_clean_cfg_mode，支持 shared/split/null
    status: completed
  - id: cli-toggle
    content: sample_pix_ddt_project_free_layer.py 增加 --s-clean-cfg-mode，写入输出 tag
    status: completed
  - id: null-s-clean-token
    content: 模型支持可学习 null_s_clean，训练时与 y dropout 同步置空，采样可用 --s-clean-cfg-mode null
    status: completed
  - id: null-ddp-fix
    content: 修复 _apply_s_clean_null_token 的早退导致 DDP "parameter did not receive grad"（micro-batch 恰好零 dropout 时）
    status: completed
  - id: intrain-eval-mode
    content: 训练内可视化采样 project_sample_ode_naive 传入 s_clean_cfg_mode，默认随模型（有 null token 则 null）
    status: completed
  - id: small-768-config
    content: 新增 6x768 / DiTDH-S / rae_inject_layers=[2] / null token 的训练+采样 config 和启动脚本（从头训）
    status: completed
  - id: train-768-null
    content: 训练 6x768 null_s_clean 到 ~ep100（当前已过 ep60，FID 仍在降）
    status: in_progress
  - id: ep60-rough-sweep
    content: ep60 上粗扫 cfg（pix_ts=1.0）和 pix_ts（cfg=1.5），定位最优区
    status: completed
  - id: ep60-round2-sweep
    content: 第二轮 sweep：cfg=2.5 固定，pix_ts 上界拉高到 6，找 pix_ts 的底
    status: pending
  - id: null-vs-shared-ablation
    content: 同 ckpt/cfg/pix_ts 下比 null vs shared(vs split)，隔离出 null_s_clean（Pixel-side CFG）的净增益
    status: pending
  - id: split-mode-sweep
    content: 方案2——uncond 用 y_null 提取的 s_clean（= 已实现的 split 模式，无需新代码），纳入对照 sweep
    status: pending
  - id: dual-traj-mode
    content: 方案1.A——新增 s_clean_cfg_mode=traj：Phase1 跑 cond/uncond 两条独立 RAE 轨迹，uncond s_clean 来自真·无条件 latent（sampling-only，任意 ckpt 可用）
    status: completed
  - id: dual-traj-sweep
    content: ep60 四模式对照已跑——traj 灾难性变差（cfg 越大越糟），根因是两条轨迹分叉成两张不同的图；null 干净胜出
    status: completed
  - id: decouple-rae-cfg
    content: 实现 --rae-cfg-scale，把 Phase1(RAE) 与 Phase3(Pixel) 的 CFG scale 解耦（默认耦合，1.0=RAE 关）
    status: completed
  - id: rae-cfg-off-sweep
    content: 跑 RAE-CFG OFF × Pixel-CFG 的 sweep（null/split），看不锐化的 latent 是否给更 in-distribution 的 s_clean、刷新最优
    status: pending
  - id: fid50k-final
    content: 在 ep100 最优 ckpt 上用最优 (cfg, pix_ts, null) 跑 FID50k，并对 baseline 同口径对照
    status: pending
isProject: false
---

# CFG s_clean 实验计划（split / null）

## 背景：三阶段串行采样里的 CFG

project / free_layer 采样是三阶段串行：

```text
Phase 1: z_rae -> RAE ODE (DiTDH 主干) -> x_rae_clean        （50 步 Euler）
Phase 2: x_rae_clean -> extract_clean_s(...) -> s_clean       （1 次前向）
Phase 3: z_pix -> Pixel ODE，条件为固定的 s_clean -> image    （50 步 Euler）
```

开 CFG 时 `z_pix`、`z_rae`、`y` 都扩成 `[cond; uncond]`：

```text
z_pix = [z_pix, z_pix]
z_rae = [z_rae, z_rae]
y     = [y_class, y_null]      # y_null = 第 1000 号空类
guided = uncond + cfg_scale * (cond - uncond)
```

`cond - uncond` 是否“干净”，直接决定 CFG 有没有用。

## 关键事实：RAE 流（Phase 1）也在做 CFG，且和 Pixel 共用同一个 scale

- `sample_pix_ddt_project_free_layer.py` 调 `project_sample_ode_naive`（从 train 脚本 import）。
- Phase 1：`if using_cfg: model.forward_rae_with_cfg(..., cfg_scale=cfg_scale, ...)` —— **RAE/DiTDH 主干带 CFG 去噪**，产出更贴类别的 `x_rae_clean` → 更好的 s_clean。
- Phase 3：`forward_pix_with_cfg(..., cfg_scale=...)` —— Pixel Head 带 CFG。

**一个 `--cfg-scale` 同时驱动两条流，当前代码不能分别调 RAE-cfg 与 Pixel-cfg。**

含义：
- 开 CFG 的 FID 收益 = **RAE-CFG + Pixel-CFG 叠加**。
- **RAE-CFG 不依赖 null token**，老 baseline 就有（Project.md 里那条“Pixel Head 开不开 CFG 信息一样，但 RAE 用了 CFG 速度流”的注释说的就是它）。
- **Pixel-CFG 才是 null_s_clean 新带来的**：没有 null token 时 uncond 半边的 s_clean 仍是 class-conditioned，Pixel 的 `cond - uncond` 被冲淡甚至近似空操作。
- 因此要**隔离 null_s_clean 的净贡献**，必须在同 ckpt / 同 cfg / 同 pix_ts 下比 `null` vs `shared`（两者 RAE-CFG 都开），差值才是 Pixel-side CFG 的增益。

## uncond 的 s_clean 怎么取：shared / split / null

Phase 2 给 uncond 半边的 s_clean 有三种取法（均已实现，`--s-clean-cfg-mode` 选择）：

| mode | uncond 的 s_clean | 含义 | 需要 null token? |
|---|---|---|---|
| `shared` | `extract_clean_s(x_rae_clean, y_class)`（复用 cond） | uncond 仍带类别语义 → Pixel-CFG 弱 | 否 |
| `split` | `extract_clean_s(x_rae_clean, y_null)` | RAE latent 仍参与，只是 DDT encoder 的 label 条件置空 | 否 |
| `null` | 可学习参数 `null_s_clean`（学到的 ∅） | Pixel Head 看到“完全没有 RAE 条件”的占位符 | **是** |

`split` vs `null` 的本质区别（也是和队友讨论的核心）：

- `split` 是**推理期**技巧，把真实 `x_rae_clean` 再过一遍 encoder、只把 label 置空。模型训练时**从没见过**这种 s_clean，多少有点 out-of-distribution。
- `null` 是**训练期**方案：给 s_clean 一个专门的、可学习的 ∅，训练时随 y dropout 同步置空，模型 in-distribution 地学到一个真正的“无条件 s_clean”，和 y_embedder 的第 1000 号空类完全对称。

## 两个 sampling 级实验：方案1（双轨 uncond）/ 方案2（split）

两个都只动采样、不动训练、不用重训。它们和 shared/split/null 一起，构成"uncond 的 s_clean 有多无条件"的一条谱：

| 模式 | uncond 的 s_clean | uncond latent | 需 null token | 状态 |
|---|---|---|---|---|
| `shared` | 复用 cond（= extract(x_guided, y_class)） | = cond（CFG-guided） | 否 | 已实现 |
| `split`（**方案2**） | extract(x_guided, **y_null**)（同一 latent，只丢类别） | = cond（CFG-guided） | 否 | **已实现** |
| `traj`（**方案1.A**） | extract(**x_uncond**, y_null)（独立无条件轨迹） | **独立无条件 latent** | 否 | **新增已实现** |
| `null` | 学到的空白 ∅ | —（不经 latent） | 是 | 已实现 |

### 方案2 = 已实现的 `split`

"空 s_clean 来自 y_null、正常 s_clean 来自 y_label"，正是 `split`：cond/uncond 用**同一个** Phase-1 guided latent，只是 encoder 的 label 一个真类、一个 null。直接 `--s-clean-cfg-mode split` 即可，无需新代码。

### 方案1 = 新增 `traj`（双轨）

Phase 1 **不做 CFG 混合**，让 `[y_class, y_null]` 两半各自演化成两个不同的干净 latent：cond 半 → conditional latent，uncond 半 → unconditional latent；再各自 extract 出 s_clean，Phase 3 正常做 Pixel CFG。

- 比 `split` 更彻底：uncond 的 **latent 和 label 都无条件**（split 是同一条件 latent + 假 null 标签，有 latent/label 失配）。
- **不依赖 null token，任意 ckpt 可用**——可在无 null token 的 baseline 上也做出 Pixel-CFG，从而和"训练出来的 null"公平对比。

**A / B 两版**

| 版本 | cond 半 latent | 每步算力 | 备注 |
|---|---|---|---|
| **A（已实现）** | 纯 conditional（forward_rae，不混） | 1×（同现在） | cond latent 不锐化；guidance 全交给 Phase 3 |
| B（未实现） | CFG 锐化（cond x 上额外算 y_null 速度） | ~1.5× | cond latent 锐度同现在，但 Phase1+Phase3 CFG 叠两层，可能过锐 |

当前实现的是 **A 版**：`s_clean_cfg_mode != "traj"` 时走原 `forward_rae_with_cfg`；`== "traj"` 时 Phase 1 改用不混合的 `forward_rae`，Phase 2 的 uncond 用 `x_rae_clean[B:]`（无条件轨迹）提取。代码点：`project_sample_ode_naive`（Phase 1 分支 + Phase 2 `traj` 分支）、`--s-clean-cfg-mode` 新增 `traj` 选项。

### 采样命令（任意已有 ckpt，含 baseline）

```bash
torchrun ... src/sample_pix_ddt_project_free_layer.py \
  --config <采样config> --ckpt <ckpt> --sample-dir <out> \
  --num-fid-samples 5000 --per-proc-batch-size 64 --precision fp32 \
  --label-sampling equal --naive \
  --cfg-scale 2.5 --s-clean-cfg-mode traj      # 方案1.A
  # 或 --s-clean-cfg-mode split                 # 方案2
```

> 注：`traj` 当前是 A 版（cond latent 不再被 Phase-1 CFG 锐化）。若 traj 反而比 shared/null 差，可能就是丢了这层锐化，再考虑实现 B 版。

## ep60 四模式对照结果（FID5k, pix_ts=1.0）

| mode | cfg 2.0 | cfg 2.5 |
|---|---|---|
| **null** | 19.92 | **19.53** |
| shared | 24.69 | 22.54 |
| split | 24.96 | 23.12 |
| traj | 37.46 | 43.31（cfg 越大越糟） |

**结论(都在同模型、RAE-CFG 都开的前提下):**
- **核心假设成立**:`null` 干净地赢 `shared`/`split` 约 **3 个 FID**。这 ~3 就是可学习 ∅(Pixel-side CFG)的净贡献——训练一个 ∅ 确实让 Pixel Head 做出了真正有效的 CFG。
- `null` 与之前 cfg sweep 一字不差 = 可复现性 sanity check 通过。
- `shared ≈ split`:非训练手段拿不到这块增益。

### traj 为什么灾难（根因，非 bug）

两个原因叠加：
1. **丢了 RAE-CFG**:traj 的 Phase 1 不做 CFG 混合,cond latent 没被锐化(A 版固有代价)。
2. **cond/uncond 是两张不同的图**(致命):同一噪声在 y_class / y_null 下去噪分叉成图 A / 图 B,于是 `s_clean_cond`(图A)与 `s_clean_uncond`(图B)描述不同的图,Pixel CFG 的 `cond−uncond` 变成"图A−图B"的无意义方向 → cfg 越大越糟(37→43)。

**B 版也救不了**:可证 `traj-B 的 cond 路 = split 的 cond 路`,唯一区别是 traj-B 的 uncond 来自另一张图(比 split 的"同图丢类别"更脏)→ `traj-B ≥ split > null`,必输。所以 traj/traj-B 是死路,根因②无解。

> 反过来这正好说明:队友想要的"好 s_clean vs 坏 s_clean 做 CFG"是对的,但"坏 s_clean"不能用另一张图来凑——它的正确形态就是训练出来的 ∅(= `null`),而 `null` 就是赢家。

## RAE-CFG 解耦（"RAE 不开 CFG"实验）

队友另一个想法:把 Phase 1(RAE)和 Phase 3(Pixel)的 CFG **解耦**,让 RAE 流可以**不开** CFG。

**理论动机(为什么有希望):** 训练时 Pixel Head 的 s_clean 来自 `extract(x1_rae, y)`,而 `x1_rae` 是**真实图像**的 DINOv2 编码(自然分布)。推理时若 RAE-CFG 开,Phase 1 的 latent 被 CFG **过度锐化**,偏离真实 latent 分布 → s_clean 对 Pixel Head 偏 OOD;**RAE-CFG 关**则得到一张自然条件 latent,s_clean 更 in-distribution,**可能反而更好**。且它**没有 traj 的"两张图"问题**(Phase 1 仍只产一张条件图,cond/uncond 对照交给 Pixel 流的 null/split)。

**实现(已完成):** `project_sample_ode_naive` 新增 `rae_cfg_scale`，CLI `--rae-cfg-scale`：
- `None`(默认)→ 耦合到 `--cfg-scale`,行为完全不变。
- `1.0` → RAE-CFG **关**:Phase 1 用 `forward_rae`、两半都喂 `y_class`(`y_phase1=[y_class,y_class]`)→ 得到**同一张自然条件 latent**(无锐化、无分叉);Pixel 流仍用 `--cfg-scale` 做 CFG。
- `> 1.0` → RAE-CFG 开,scale 独立于 Pixel。
- `traj` 模式忽略此参数(它自成一路)。

**待跑 sweep:** `cfg_sweep_null768_ep60.sh` 已加第 (2) 段——`null/split` 模式 × `rae_cfg=1.0(关)` × `pix_cfg ∈ {1.5,2.0,2.5,3.0}`。和第 (1) 段(RAE-CFG 开)同表对比,看关掉 RAE-CFG 能否把 19.5 再压低。

## null_s_clean 的具体实现

### 模型参数（`src/stage2/models/PixelDDTv2_project_free_layer.py`）

```python
# __init__，仅当 use_s_clean_null_token=true
self.null_s_clean = nn.Parameter(torch.zeros(
    1, self.ddt.s_embedder.num_patches, self.ddt.encoder_hidden_size
))   # DiTDH-S 下 = [1, 256, 384]
```

**关于全0：那是“初始化”，不是固定的全局常量。** `torch.zeros(...)` 只在建模型时跑一次给个起点；`nn.Parameter` 默认 `requires_grad=True`，所以它是**可训练**的，会被优化器一路更新、逐渐离开 0（类似 AdaLN-Zero 的零初始化思路）。

可训练性的证据链：
- 不像 `pos_embed` 显式写了 `requires_grad=False`，`null_s_clean` 没关梯度 → 默认 True。
- 唯一被冻结的是 DDT 主干：`for p in self.ddt.parameters(): p.requires_grad = False`，`null_s_clean` 不在 `self.ddt` 里，不受影响。
- 优化器 `build_optimizer([p for p in model.parameters() if p.requires_grad], ...)` 会收它。
- 采样时 `get_null_s_clean()` 返回的是 checkpoint 里**学好的**值（不是 0）。

### 训练 forward（同步置空）

```python
force_drop_ids = self._sample_cfg_drop_ids(y)        # 采样一次，下面三处共用
s_clean = self._extract_clean_s(x1_rae, y, force_drop_ids=force_drop_ids)
s_clean = self._apply_s_clean_null_token(s_clean, force_drop_ids)
imgs_out = self.forward_pix(x_pix, t_pix, y, s_clean, force_drop_ids=force_drop_ids)
```

同一个 `force_drop_ids` 同时驱动：① extract encoder 里的 y-drop；② s_clean → null_s 的整体替换；③ Pixel Head 里 `y_embedder` 的 y-drop。**三者严格同步**，所以 uncond 分支同时没有类别标签也没有 RAE 语义条件，接近真正的无条件分支。

`_apply_s_clean_null_token`（整条 s_clean 被替换，按样本）：

```python
def _apply_s_clean_null_token(self, s_clean, force_drop_ids):
    if not self.use_s_clean_null_token or force_drop_ids is None:
        return s_clean
    drop_mask = (force_drop_ids == 1).view(-1, 1, 1)
    null_s = self.get_null_s_clean(s_clean.shape[0], ...)
    return torch.where(drop_mask, null_s, s_clean)   # 整条 [L,C] 换成 ∅
```

### DDP（已修复）

旧实现里有一行早退：

```python
if not drop_mask.any():
    return s_clean        # ← bug：这个 micro-batch 没人被 drop 时直接返回
```

这会让 `null_s_clean` 在“恰好零 dropout”的 micro-batch 里**不进计算图 → 拿不到梯度**，DDP 的 reducer 等不到它，下一步报：

```text
RuntimeError: Expected to have finished reduction in the prior iteration ...
Parameter indices which did not receive grad ...
```

dropout=0.1、micro-batch=32 时约 3.4% 的 micro-batch 会触发，所以训练能跑过 step 0、几十步后才崩。**修复 = 删掉早退，永远走 `torch.where`**：drop_mask 全 False 时输出仍等于 s_clean（行为不变），但 `null_s` 每步都在图里、拿到（全零）梯度，DDP 不再报错。零开销，无需 `find_unused_parameters`。

### 一个无害的冗余

`_extract_clean_s(x1_rae, y, force_drop_ids=...)` 对被 drop 的样本会先在 encoder 里把 y 置空，紧接着第二行又被 ∅ 整体覆盖——对 `null` 模式这步 y-drop 是冗余的（结果被覆盖），但**必须保留**：`use_s_clean_null_token=false` 的 shared/split 模式靠它做 dropout。冗余但输出无差别。

### 采样侧

`get_null_s_clean(B)` 把 `[1,L,C]` 扩到 `[B,L,C]`；`--s-clean-cfg-mode null` 时 Pixel CFG 两路为：

```text
cond:   y_class, s_clean_cond
uncond: y_null,  learned_null_s_clean
```

`project_sample_ode_naive(s_clean_cfg_mode=...)` 三模式分支都在；训练内的可视化采样也会传 `s_clean_cfg_mode`（默认随模型：有 null token 则默认 `null`，否则 `shared`）。

> 旧 checkpoint 没有 `null_s_clean` 参数，不能用 `--s-clean-cfg-mode null`（会缺参数）。

## 实际跑的实验（当前主线）：6x768 小模型从头训 null_s_clean

> 历史原因：1152 大模型训得太慢，先在小模型上验证 null_s_clean 的想法。

- 架构：`hidden_size=768, depth=6, num_heads=12`，backbone DiTDH-S；`rae_inject_layers=[2]`（对齐文档里 6 层 baseline，便于对照）；`rae_inject_final_layer=false`、`use_y_embedder=true`、`use_adaln_zero=true`、`use_lora=false`、`use_s_clean_null_token=true`。
- 时间调度：`pix_time_shift=1.0`（misc dim 4096）、`rae_time_shift≈6.93`（misc_rae dim 196608）；`logit-normal_0.8_0.8`；lr 2e-4；**从头训**（不传 --ckpt，只加载冻结 DiTDH-S 主干）。trainable ≈ 67M。
- 文件：
  - 训练 config：`configs/stage2/training/PixelDDT_free_layer_2_wo_lora_w_y_embed_adaln_zero_6_768_DiTDHS_null_s_clean.yaml`
  - 采样 config：`configs/stage2/sampling/ImageNet256/PixDDT_free_layer_2_wo_lora_w_y_embed_adaln_zero_6_768_DiTDHS_null_s_clean.yaml`
  - 启动脚本：`retry_train_stage3_project_free_layer_null_s_clean_768.sh`
- EXPERIMENT_NAME 仍是旧的 1152 名字（切小模型时漏改），目录名只是标签，**实际权重是 6x768 [2] null（67M，已用 config.yaml 快照核对）**。不影响训练，等训完再改目录。

## 结果（ep60，FID5k，ref=VIRTUAL_imagenet256_labeled.npz）

训练内自动 FID（cfg=1.5, pix_ts=1.0, mode=null）：

```text
ep10 37.87 | ep20 27.79 | ep30 25.94 | ep40 25.89 | ep50 24.67 | ep60 23.83
```

ep60 CFG sweep（pix_ts=1.0, null）：

```text
scale 1.0(无CFG) 42.21 | 1.5 23.83 | 2.0 19.92 | 2.5 19.53(最优) | 3.0 19.86
```

ep60 pix_ts sweep（cfg=1.5, null）：

```text
0.2 25.10 | 0.3 24.78 | 0.5 24.47 | 0.8 24.10 | 1.0 23.83
1.5 23.06 | 2.0 22.33 | 2.5 21.82 | 3.0 21.51（到上界仍在降）
```

观察：
- CFG 是大杠杆（42 → ~19.5），最优 scale ≈ 2.5（2.0–3.0 是平谷）。
- pix_ts 单调下降、到 3.0 还没见底，且这是在次优 cfg=1.5 下扫的 → 需要在 cfg=2.5 下、把 pix_ts 上界拉高再扫一轮。
- 初步对照 **看起来 null_s_clean 的 Pixel-side CFG 带来明显增益**。但 epoch 未完全对齐、FID5k 有噪声，结论待严格对照确认。

## 工具

- `cfg_sweep_null768_ep60.sh`：基于 `cfg_split_s_clean_sweep.sh`，固定 ep60 ckpt + `_2_` 采样 config，`--s-clean-cfg-mode null` 扫 cfg scale。`run_combo MODE SCALE T_MIN T_MAX` 第一参数 mode，可顺手加 `shared/split` 做对照。
- `eval_pix_ts_sweep.py`：按 `dim = ts²×4096` patch config 扫 pix_ts。**已补上原脚本缺的两个开关** `CFG_SCALE` 和 `S_CLEAN_CFG_MODE`（之前不传 → 默认变成无 CFG + shared，对 null 模型不对）。
  - ⚠️ 它的 npz 只按 pix_ts 命名（不含 cfg）。**换 cfg 重扫时必须换 `EXPERIMENT_ROOT`/`SAMPLE_ROOT`**，否则会 skip 采样、复用旧 cfg 的 npz、报出 stale 数字。

## 下一步

1. **第二轮 sweep**：`CFG_SCALE=2.5`、`PIXEL_TIME_SHIFTS=[1.0,2.0,3.0,4.0,5.0,6.0]`、新输出目录 `~/data/samples/null768_ep60_cfg2.5`，找 pix_ts 的底。扫完 ep60 不再多扫。
2. **null vs shared(vs split) ablation**：同 ckpt / 同 cfg(2.5) / 同 pix_ts，比三种 mode，隔离出 null_s_clean（Pixel-CFG）的净增益。
3. **续训到 ep100**（resume 自动从最新 ckpt 接续，不会从头）。
4. **FID50k 终评**：在 ep100 最优 ckpt 上用最优 (cfg, pix_ts, null) 跑 FID50k，并对 baseline 同口径对照，给出最终结论。

## 判读规则（针对 null vs baseline）

- **null 在最优 cfg 处明显优于 shared/baseline**：说明可学习 ∅ 让 Pixel Head 真正能 CFG，验证想法 → 推进 ep100 + FID50k。
- **null ≈ shared**：说明增益主要来自 RAE-CFG，Pixel-side CFG 没帮上忙 → 重新审视注入层/训练是否让 Pixel 真正用到了 s_clean。
- **null 反而更差**：可能 ∅ 没学好或注入层太少（[2] 只注 1 层），可考虑增加注入层或更长训练再评。
