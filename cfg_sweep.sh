#!/bin/bash
# CFG 参数组合扫描脚本 (ep-0000125, FID5k)
# 18 种组合：全范围 scale 扫描 + t_min/t_max 部分范围变体
# 不修改 yaml，通过 CLI 参数和 SAVE_FOLDER 环境变量控制
# 运行方式：bash cfg_sweep.sh（从 RAE/ 根目录运行）

set -e

CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/PixDDTv2_fp32/checkpoints/ep-0000125.pt}
CONFIG=${CONFIG:-/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixelDDTv2_3.yaml}
SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/cfg_sweep_ep125}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
NPROC=${NPROC:-4}

if [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found: $CKPT" >&2
    echo "        Locate the checkpoint, then rerun with:" >&2
    echo "        CKPT=/path/to/ep-0000125.pt bash cfg_sweep.sh" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 1
fi
if [ ! -f "$REF" ]; then
    echo "[error] reference stats not found: $REF" >&2
    exit 1
fi

mkdir -p "$SAMPLE_BASE"
RESULTS_FILE="$SAMPLE_BASE/fid_results.txt"

# 追加分隔线，不覆盖已有结果
echo ""                                                                >> "$RESULTS_FILE"
echo "--- Round 2 remainder: fine-tune scale & t_min near optimum ---" >> "$RESULTS_FILE"
printf "%-8s  %-8s  %-8s  %s\n" "scale" "t_min" "t_max" "FID5k"      >> "$RESULTS_FILE"
echo "----------------------------------------"                        >> "$RESULTS_FILE"

# ---------- 核心函数 ----------
run_combo() {
    local scale=$1
    local t_min=$2
    local t_max=$3

    # 文件夹/npz 命名：cfg-s{scale}-tmin{t_min}-tmax{t_max}
    local tag
    tag=$(printf "cfg-s%.1f-tmin%.1f-tmax%.1f" "$scale" "$t_min" "$t_max")
    local npz_path="$SAMPLE_BASE/${tag}.npz"

    echo ""
    echo "============================================================"
    echo "  scale=$scale  t_min=$t_min  t_max=$t_max  tag=$tag"
    echo "============================================================"

    # 如果 npz 已存在则跳过采样
    if [ -f "$npz_path" ]; then
        echo "  [SKIP] npz already exists: $npz_path"
    else
        SAVE_FOLDER="$tag" \
        torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
            src/sample_pix_ddt.py \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --sample-dir "$SAMPLE_BASE" \
            --num-fid-samples 5000 \
            --per-proc-batch-size 64 \
            --precision fp32 \
            --label-sampling equal \
            --naive \
            --cfg-scale "$scale" \
            --t-min "$t_min" \
            --t-max "$t_max"
    fi

    echo "  Computing FID..."
    fid_out=$(cd /home/shihaoyang/RAE/src && python -m eval.fid \
        --arr1 "$npz_path" \
        --ref "$REF" \
        --ref-mu-key mu \
        --ref-sigma-key sigma 2>&1)
    # 取最后一行（FID 数值那行）
    fid_val=$(echo "$fid_out" | tail -1)
    echo "  -> FID: $fid_val"
    printf "%-8s  %-8s  %-8s  %s\n" "$scale" "$t_min" "$t_max" "$fid_val" >> "$RESULTS_FILE"
}

# =====================================================================
# Round 2：11 种精细组合
# 依据 Round 1 结论：
#   · t_max 必须 = 1.0（截断精细去噪段会导致 FID 暴涨）
#   · 最优 scale ≈ 2.5，t_min ≈ 0.3 有轻微增益
#   · 目标：在 scale ∈ [2.2, 2.8] 和 t_min ∈ [0.0, 0.5] 精细搜索
# =====================================================================

# --- 组 E：scale 精细步长（全范围，t_min=0.0），0.1 步长 ---
# 已知 2.5 最优，检查 2.3/2.4/2.6 是否更好
run_combo 2.3  0.0 1.0
run_combo 2.4  0.0 1.0
run_combo 2.6  0.0 1.0

# --- 组 F：t_min 精细调（scale=2.5，t_max=1.0 固定） ---
# 0.3 略优于 0.0，继续在 [0.1, 0.5] 内细化
run_combo 2.5  0.1 1.0
run_combo 2.5  0.2 1.0
run_combo 2.5  0.4 1.0
run_combo 2.5  0.5 1.0

# --- 组 G：交叉最优候选（最优 scale × 最优 t_min 附近） ---
# 将精细 scale 与精细 t_min 组合，找真正最优交点
run_combo 2.3  0.3 1.0
run_combo 2.4  0.3 1.0
run_combo 2.6  0.3 1.0
run_combo 2.4  0.2 1.0

# =====================================================================
echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
