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
# I：围绕当前最优 scale=1.8,t_min=0,t_max=1 做精细扫描
# 已知：
#   · t_min>0 在粗扫中普遍变差，但仍轻扫 0.03/0.05/0.08 排除小 interval 收益
#   · t_max<1 在旧模型上很差，这里只轻扫 0.95/0.98，不碰 0.7/0.8
#   · FID5k 最优当前为 scale=1.8,t_min=0,t_max=1；FID50k 历史记录 scale=2
# =====================================================================

# --- 旧 E/F/G/H 组保留但暂停，避免重复消耗 ---
# run_combo 2.3  0.0 1.0
# run_combo 2.4  0.0 1.0
# run_combo 2.6  0.0 1.0
# run_combo 2.5  0.1 1.0
# run_combo 2.5  0.2 1.0
# run_combo 2.5  0.4 1.0
# run_combo 2.5  0.5 1.0
# run_combo 2.3  0.3 1.0
# run_combo 2.4  0.3 1.0
# run_combo 2.6  0.3 1.0
# run_combo 2.4  0.2 1.0
# run_combo 1.8   0.0 1.0
# run_combo 2.0   0.0 1.0
# run_combo 2.1   0.0 1.0
# run_combo 2.2   0.0 1.0
# run_combo 2.25  0.0 1.0
# run_combo 2.3   0.0 1.0
# run_combo 2.35  0.0 1.0

# --- 组 I1：scale around 1.8, full interval ---
run_combo 1.50  0.0 1.0
run_combo 1.60  0.0 1.0
run_combo 1.70  0.0 1.0
run_combo 1.75  0.0 1.0
run_combo 1.80  0.0 1.0
run_combo 1.85  0.0 1.0
run_combo 1.90  0.0 1.0
run_combo 1.95  0.0 1.0

# --- 组 I2：tiny t_min gating around the current best scale ---
run_combo 1.70  0.03 1.0
run_combo 1.80  0.03 1.0
run_combo 1.90  0.03 1.0
run_combo 1.70  0.05 1.0
run_combo 1.80  0.05 1.0
run_combo 1.90  0.05 1.0
run_combo 1.80  0.08 1.0

# --- 组 I3：very light t_max check, only near full interval ---
run_combo 1.80  0.0 0.98
run_combo 1.80  0.0 0.95

# =====================================================================
echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
