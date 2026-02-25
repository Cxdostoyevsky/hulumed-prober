import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

# ================= 配置区域 =================
# 1. 原始底座模型路径
BASE_MODEL_PATH = "/ssd/common/LLMs/Hulu-Med-4B"

# 2. 训练好的 LoRA 路径 (包含 adapter_model.safetensors 的文件夹)
ADAPTER_PATH = "/ssd/common/LLMs/Hulu-Med-4B_finetune/normal/final_model"

# 3. 合并后的模型保存路径 (你可以修改这个)
OUTPUT_PATH = "/ssd/common/LLMs/Hulu-Med-4B_finetune/normal/Hulumed-4B-merged"


# ===========================================

def merge_model():
    print(f"🚀 开始合并模型...")
    print(f"📂 Base Model: {BASE_MODEL_PATH}")
    print(f"📂 Adapter:    {ADAPTER_PATH}")

    # 1. 加载 Base Model
    # 注意：合并时必须以非量化形式加载 (float16 或 bfloat16)，不能用 load_in_4bit/8bit
    print("\n⏳ 正在加载 Base Model (这可能需要一点时间)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,  # 建议使用 float16 节省显存
            device_map="auto",  # 自动分配显卡或 CPU
            trust_remote_code=True  # 允许加载自定义模型代码
        )
    except Exception as e:
        print(f"❌ 加载 Base Model 失败: {e}")
        return

    # 2. 加载 Tokenizer
    print("⏳ 正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ 加载 Tokenizer 警告 (可能需要手动复制): {e}")
        tokenizer = None

    # 3. 加载 LoRA Adapter
    # 这一步会自动处理 lora 权重合并以及 modules_to_save (mm_projector) 的替换
    print("⏳ 正在加载 LoRA Adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except Exception as e:
        print(f"❌ 加载 Adapter 失败: {e}")
        return

    # 4. 执行合并
    print("\n🔄 正在执行 merge_and_unload (合并权重)...")
    model = model.merge_and_unload()

    # 5. 保存合并后的模型
    print(f"💾 正在保存合并后的模型到: {OUTPUT_PATH}")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # 先保存新生成的权重和配置
    model.save_pretrained(OUTPUT_PATH)

    if tokenizer:
        tokenizer.save_pretrained(OUTPUT_PATH)
        print("✅ Tokenizer 已保存")
    else:
        print("⚠️ 请记得手动复制 tokenizer 相关文件到输出目录")

    # 6. 从 Base Model 复制其他文件并覆盖配置
    print("\n📦 正在从 Base Model 复制文件并覆盖非权重文件...")
    
    # 遍历 Base Model 目录下的所有文件
    for item in os.listdir(BASE_MODEL_PATH):
        src_path = os.path.join(BASE_MODEL_PATH, item)
        dst_path = os.path.join(OUTPUT_PATH, item)
        
        # 核心逻辑：如果是权重文件（.safetensors 或 .index.json），跳过复制，保留新生成的
        if item.endswith(".safetensors") or item == "model.safetensors.index.json":
            continue
            
        # 其他所有文件（包括 config.json, tokenizer.model 等），强制从 Base 复制并覆盖
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  - 已覆盖/复制文件: {item}")
        elif os.path.isdir(src_path):
            # 如果有子文件夹（如 vision_encoder），递归复制
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            print(f"  - 已覆盖/复制目录: {item}")

    print(f"\n🎉 合并及文件整理完成！你可以直接使用 {OUTPUT_PATH} 进行推理了。")


if __name__ == "__main__":
    merge_model()
