#!/bin/bash

# ===================================================
# Hulu-Med BUSI 乳腺癌分类微调启动脚本
# ===================================================


## 1. 多卡训练（如果有多张 GPU，取消下面的注释）
# echo "🚀 开始多卡训练（使用 DeepSpeed ZeRO-2）..."
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_busi_lora.py

# 2. 如果显存不够，可以启用 QLoRA（4bit 量化）
# 修改 train_busi_lora.py 中的 `use_qlora=True`





# 3. 单卡训练（推荐显存 >= 24GB）
#echo "🚀 开始单卡训练..."
#CUDA_VISIBLE_DEVICES=7 python train_busi_lora.py

# 4.开启探针模模式
# - `--probe_position`：指定探针位置（如 vision_encoder_10, llm_5, projector)
# - `--probe_type`：指定探针类型（如 mlp, linear)
# 视觉编码器和语言模型的探针的维度不同，由传入的参数自行判断(Vision:1152,llm:2560)，无需用户指定
# 范围：vision_encoder_0 ~ vision_encoder_26, llm_0 ~ llm_35, projector
#echo "🚀 开始探针模训练..."
#CUDA_VISIBLE_DEVICES=7 python train_busi_lora.py \
#  --probe_position vision_encoder_10 \
#  --probe_type mlp \


# ===================================================
# 批量训练探针脚本
# ===================================================

# 定义要训练的所有探针位置
POSITIONS=(
  "vision_encoder_1"
  "vision_encoder_3"
  "vision_encoder_5"
  "vision_encoder_7"
  "vision_encoder_19"
  "vision_encoder_11"
  "vision_encoder_13"
  "vision_encoder_15"
  "vision_encoder_17"
  "vision_encoder_19"
  "vision_encoder_21"
  "vision_encoder_23"
  "vision_encoder_25"
  "projector"
  "llm_1"
  "llm_10"
  "llm_12"
  "llm_14"
  "llm_16"
  "llm_25"
  "llm_35"
)

PROBE_TYPE="mlp"
GPU_ID=7

echo "🚀 开始批量训练探针..."

for pos in "${POSITIONS[@]}"; do
  echo "---------------------------------------------------"
  echo "▶️  正在训练探针: position=${pos}, type=${PROBE_TYPE}"
  echo "---------------------------------------------------"

  CUDA_VISIBLE_DEVICES=${GPU_ID} python train_busi_lora.py \
    --probe_position "${pos}" \
    --probe_type "${PROBE_TYPE}" \

  if [ $? -ne 0 ]; then
    echo "❌ 训练失败: ${pos}"
    exit 1
  fi

  echo "✅ 完成: ${pos}"
  echo ""
done

echo "🎉 所有探针训练完成！"
