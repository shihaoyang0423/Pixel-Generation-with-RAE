#!/bin/bash
# CFG 参数组合扫描脚本 (当前 free_layer 主力配置, FID5k)
# 扫描 CFG scale 与 t_min/t_max，不修改 yaml，通过 CLI 参数和 SAVE_FOLDER 控制
# 不修改 yaml，通过 CLI 参数和 SAVE_FOLDER 环境变量控制
# 运行方式：bash cfg_sweep.sh（从 RAE/ 根目录运行）

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-PixDDT_small_v2_project_free_layer_1_3_5_7_false_wo_lora_w_y_embed_adaln_zero_12_768}
CKPT=${CKPT:-/home/shihaoyang/data/ckpts/PixDDTv2/${EXPERIMENT_NAME}/checkpoints/ep-0000050.pt}
CONFIG=${CONFIG:-/home/shihaoyang/RAE/configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project_free_layer.yaml}
SAMPLE_BASE=${SAMPLE_BASE:-/home/shihaoyang/data/samples/${EXPERIMENT_NAME}/cfg_sweep}
REF=${REF:-/home/shihaoyang/data/data/VIRTUAL_imagenet256_labeled.npz}
NPROC=${NPROC:-8}

if [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found: $CKPT" >&2
    echo "        Locate the checkpoint, then rerun with:" >&2
    echo "        CKPT=/path/to/ep-0000050.pt bash cfg_sweep.sh" >&2
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
echo "--- free_layer CFG sweep: ${EXPERIMENT_NAME} ---"                >> "$RESULTS_FILE"
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
            src/sample_pix_ddt_project_free_layer.py \
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
# 当前 free_layer 链路先复现/围绕 CFG scale=2.0 搜索：
#   · t_max 固定 1.0，避免截断后段去噪
#   · 先扫 scale，再轻扫 t_min
# =====================================================================

# --- 组 A：复现无 CFG / 日志中 scale=2 附近 ---
run_combo 1.0  0.0 1.0
run_combo 1.6  0.0 1.0
run_combo 1.8  0.0 1.0
run_combo 2.0  0.0 1.0
run_combo 2.2  0.0 1.0
run_combo 2.4  0.0 1.0

# --- 组 B：围绕 scale=2 轻扫 t_min ---
run_combo 1.8  0.1 1.0
run_combo 2.0  0.1 1.0
run_combo 2.2  0.1 1.0
run_combo 1.8  0.2 1.0
run_combo 2.0  0.2 1.0
run_combo 2.2  0.2 1.0

# =====================================================================
echo ""
echo "========================================"
echo "ALL DONE. Results appended to: $RESULTS_FILE"
echo "========================================"
cat "$RESULTS_FILE"
