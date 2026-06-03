#!/bin/bash
# CFG 参数组合扫描脚本 (free-layer 当前主力配置, FID5k)
# E/F/G 三组：scale 精扫 + t_min/t_max 部分范围变体
# 不修改 yaml，通过 CLI 参数和 SAVE_FOLDER 环境变量控制
# 运行方式：bash cfg_sweep.sh（从 RAE/ 根目录运行）

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_1152_DiTDH-XL}
CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints/ep-0000050.pt}
CONFIG=${CONFIG:-/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml}
SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXPERIMENT_NAME}/cfg_sweep}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-5000}
PRECISION=${PRECISION:-fp32}

if [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found: $CKPT" >&2
    echo "        Set CKPT=/path/to/your/free_layer_checkpoint.pt and rerun." >&2
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
echo "--- free-layer CFG sweep: ${EXPERIMENT_NAME} ---"               >> "$RESULTS_FILE"
echo "ckpt: $CKPT"                                                    >> "$RESULTS_FILE"
echo "config: $CONFIG"                                                >> "$RESULTS_FILE"
printf "%-8s  %-8s  %-8s  %s\n" "scale" "t_min" "t_max" "FID${NUM_FID_SAMPLES}" >> "$RESULTS_FILE"
echo "----------------------------------------"                        >> "$RESULTS_FILE"

# ---------- 核心函数 ----------
run_combo() {
    local scale=$1
    local t_min=$2
    local t_max=$3

    # 文件夹/npz 命名：cfg-s{scale}-tmin{t_min}-tmax{t_max}
    local tag
    tag="cfg-s${scale}-tmin${t_min}-tmax${t_max}"
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
            src/sample_pix_ddt_project_free_layer.py \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --sample-dir "$SAMPLE_BASE" \
            --num-fid-samples "$NUM_FID_SAMPLES" \
            --per-proc-batch-size "$BATCH_SIZE" \
            --precision "$PRECISION" \
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
# E/F/G/H：先保留旧 E/F/G，再围绕当前最优点加 H 组
# 依据旧 CFG sweep 结论：
#   · t_max 必须 = 1.0（截断精细去噪段会导致 FID 暴涨）
#   · 旧模型最优 scale ≈ 2.5，t_min ≈ 0.3 有轻微增益
# 新信息：
#   · 12x1152 / DiTDH-XL / pix_time_shift=3.0 的 FID50k 最好记录在 scale=2
#   · FID5k 粗扫中 scale=2.3,t_min=0.0 最好，t_min>0 普遍变差
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

# --- 组 H：围绕当前信息的主搜索中心（t_min=0.0，t_max=1.0） ---
# 目标：复现日志 scale=2，并细化 FID5k 最优 scale≈2.3。
run_combo 1.8   0.0 1.0
run_combo 2.0   0.0 1.0
run_combo 2.1   0.0 1.0
run_combo 2.2   0.0 1.0
run_combo 2.25  0.0 1.0
run_combo 2.3   0.0 1.0
run_combo 2.35  0.0 1.0

# =====================================================================
echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
