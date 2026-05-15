---
name: Pixel TS FID Sweep
overview: 写一个 Python 脚本，自动遍历用户指定的 `pixel_time_shift` 值列表，对 `ep-0000050.pt` 逐一进行 FID5k 采样和计算，将结果汇总到一个 txt 文件。
todos:
  - id: write-sweep-script
    content: 创建 /home/shihaoyang/RAE/eval_pix_ts_sweep.py，包含用户配置区、临时YAML生成、torchrun采样、FID计算、结果写入逻辑
    status: completed
isProject: false
---

# Pixel Time Shift FID5k Sweep 脚本

## 核心机制

`time_dist_shift_pix` 由 YAML 中两个参数计算而来（见 [`configs/.../PixDDT_small_v2_lora_project_free_layer.yaml`](RAE/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml) 第 84-88 行）：

```
time_dist_shift_pix = sqrt(time_dist_shift_dim / time_dist_shift_base)
```

当前配置 `dim=16384, base=4096` → `shift_pix = 2.0`。

要测试不同的 `pixel_time_shift` 值，只需反推 `dim = pixel_time_shift² × base`，动态生成临时 YAML，无需手写多个配置文件。

## 新建脚本

**文件路径：`/home/shihaoyang/RAE/eval_pix_ts_sweep.py`**

### 顶部用户配置区（用户直接修改这里）

```python
PIXEL_TIME_SHIFTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # <-- 修改这里

CKPT      = "/home/shihaoyang/data/ckpts/PixDDTv2/PixDDT_small_v2_project_free_layer_2_false/checkpoints/ep-0000050.pt"
BASE_CONFIG = "configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml"
SAMPLE_ROOT = os.path.expanduser("~/data/samples/PixDDT_small_v2_project_free_layer_2_false/compare_different_pix_ts")
REF_NPZ     = os.path.expanduser("~/data/data/VIRTUAL_imagenet256_labeled.npz")
NPROC_PER_NODE = 4
BATCH_SIZE     = 64
NUM_FID_SAMPLES = 5000
PRECISION      = "fp32"
```

### 每个 pixel_time_shift 的执行流程

```
对每个 ts 值:
  1. 计算 new_dim = int(ts² × 4096)
  2. 用 PyYAML 加载 BASE_CONFIG，修改 misc.time_dist_shift_dim，写入临时文件
  3. 设置 SAVE_FOLDER=pix_ts{ts:.2f} 环境变量
  4. 调用 torchrun 运行 sample_pix_ddt_project_free_layer.py
     → PNG 目录: {SAMPLE_ROOT}/pix_ts{ts:.2f}/
     → NPZ 文件: {SAMPLE_ROOT}/pix_ts{ts:.2f}.npz
  5. 用 PYTHONPATH=src subprocess 调用 python -m eval.fid 计算 FID
  6. 将结果追加写入 {SAMPLE_ROOT}/results.txt
```

### 输出文件命名

- PNG 目录：`compare_different_pix_ts/pix_ts2.00/`（由 `SAVE_FOLDER` 控制）
- NPZ 文件：`compare_different_pix_ts/pix_ts2.00.npz`（sampling 脚本自动生成）
- 结果汇总：`compare_different_pix_ts/results.txt`，格式示例：
  ```
  [2026-05-13 11:30:00] pix_ts=2.00  dim=16384  FID: 43.50
  [2026-05-13 11:45:00] pix_ts=1.00  dim=4096   FID: 51.23
  ```

### 脚本行为细节

- 若某个 `pix_ts{ts:.2f}.npz` 已存在，**跳过采样**，直接复用（断点续跑）
- 采样和 FID 的 stdout/stderr 同时打印到终端并 tee 到 `{SAMPLE_ROOT}/sample_{ts:.2f}.log`
- 脚本工作目录固定为 `RAE/`（用 `os.chdir` 保证相对路径正确）
- 使用 `/home/shihaoyang/miniconda3/envs/rae/bin/python` 保证 FID 计算的依赖正确

## 使用方法

```bash
cd ~/RAE
python eval_pix_ts_sweep.py
```
