---
name: Direct RAE Injection Mode
overview: 在 PixelDDT free_layer 模型中新增一个"直接注入"模式：通过 YAML 字段控制，训练/推理时可选择跳过 12 层 DDT 编码器瓶颈，直接用干净的 DINOv2 特征（reshape + 轻量线性投影）作为 Pixel Head 的条件注入。训练/采样脚本无需修改。
todos:
  - id: model-changes
    content: 修改 PixelDDTv2_project_free_layer.py：__init__ 加参数、创建 s_raw_projector、_extract_clean_s 加分支
    status: completed
  - id: yaml-train
    content: 训练 YAML PixelDDTv2_small_lora_project_free_layer.yaml 加两个新字段
    status: completed
  - id: yaml-sample
    content: 采样 YAML PixDDT_small_v2_lora_project_free_layer.yaml 加两个新字段
    status: completed
isProject: false
---

# Direct RAE Injection Mode 实现计划

## 核心差异

```
当前（瓶颈注入）:  x1_rae [B,768,16,16] → 12层DDT编码器 → s_clean [B,256,384] → s_projector → 注入
新增（直接注入）:  x1_rae [B,768,16,16] → reshape+s_raw_projector → s_clean [B,256,384] → s_projector → 注入
```

两条路输出形状相同，`forward_pix` / `project_sample_ode_naive` / 训练脚本全部无需改动。

## 文件改动

### 1. [`RAE/src/stage2/models/PixelDDTv2_project_free_layer.py`](RAE/src/stage2/models/PixelDDTv2_project_free_layer.py)

**`__init__` 新增两个参数：**
```python
use_direct_rae_injection: bool = False,
direct_rae_in_channels: int = 768,   # channel dim of x1_rae (DINOv2-base = 768)
```

**存储 flag，并在 `s_projector` 之后、`initialize_weights()` 之前创建新投影层：**
```python
self.use_direct_rae_injection = use_direct_rae_injection
self.s_projector = nn.Linear(self.ddt.encoder_hidden_size, self.hidden_size)
if use_direct_rae_injection:
    self.s_raw_projector = nn.Linear(direct_rae_in_channels, self.ddt.encoder_hidden_size)
```

**修改 `_extract_clean_s`，增加直接路径分支：**
```python
def _extract_clean_s(self, x1_rae, y):
    if self.use_direct_rae_injection:
        # 直接注入：reshape [B,C,H,W] → [B,H*W,C]，然后线性投影到 encoder_hidden_size
        B, C, H, W = x1_rae.shape
        x1_flat = x1_rae.flatten(2).permute(0, 2, 1)   # [B, H*W, C]
        return self.s_raw_projector(x1_flat)            # [B, H*W, encoder_hidden_size]
    else:
        # 原路径：过 12 层 DDT 编码器
        B = x1_rae.shape[0]
        t_zero = torch.zeros(B, device=x1_rae.device, dtype=x1_rae.dtype)
        ddt_ctx = contextlib.nullcontext() if self.use_lora else torch.no_grad()
        with ddt_ctx:
            s, _, _ = self._run_ddt_encoder(x1_rae, t_zero, y)
        return s
```

**打印信息中加上模式提示：**
```python
print(f"[PixelDDT free_layer] rae_inject_layers={rae_inject_layers}, "
      f"rae_inject_final_layer={rae_inject_final_layer}, "
      f"use_direct_rae_injection={use_direct_rae_injection}")
```

不需改动：`extract_clean_s`、`forward_pix`、`forward`、CFG 变体——它们都通过 `_extract_clean_s`/`extract_clean_s` 获取 `s_clean`，输出形状不变。

### 2. [`RAE/configs/stage2/training/PixelDDTv2_small_lora_project_free_layer.yaml`](RAE/configs/stage2/training/PixelDDTv2_small_lora_project_free_layer.yaml)

在 `stage_2.params` 末尾新增：
```yaml
    use_direct_rae_injection: false   # true → 跳过DDT编码器，直接用干净DINOv2特征
    direct_rae_in_channels: 768       # x1_rae 的通道数 (DINOv2-base)
```

### 3. [`RAE/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml`](RAE/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml)

同上，在 `stage_2.params` 末尾新增相同两行。

## 不需要改的文件

- `train_stage3_projcet_free_layer.py`：`project_training_losses` 已将 `x1_rae` 传入 `model.forward()`，`model.forward()` 内部调用 `_extract_clean_s` 自动走对应分支
- `sample_pix_ddt_project_free_layer.py`：`project_sample_ode_naive` 调用 `model.extract_clean_s(x_rae_clean, y)`，内部自动走对应分支

## 使用方式

训练新实验时，把 YAML 里 `use_direct_rae_injection` 改为 `true`，`direct_rae_in_channels` 保持 `768`，`rae_inject_layers` 仍然控制注入哪几层。
