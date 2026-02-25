"""
Hulu-Med 乳腺癌分类微调训练脚本 (LoRA/QLoRA)
支持多卡训练、梯度累积、DeepSpeed 加速
"""
import sys
import os
# [重要] 在导入 bitsandbytes 之前设置：若 PyTorch 为 CUDA 12.8 而 bitsandbytes 无预编译 128，
# 可强制使用 12.4 的二进制（需在 import 任何会触发 peft/bitsandbytes 的包之前）
if "BNB_CUDA_VERSION" not in os.environ:
    os.environ["BNB_CUDA_VERSION"] = "124"
sys.path.append(os.path.join(os.getcwd(), "src"))
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from PIL import Image

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
)


try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as e:
    raise RuntimeError(
        "导入 peft 失败，通常是 bitsandbytes/triton 与当前 CUDA/PyTorch 不兼容导致。\n"
        "如果你当前不用 QLoRA（use_qlora=False），可直接执行：\n"
        "  pip uninstall -y bitsandbytes triton\n"
        "若要使用 QLoRA，请安装与当前 PyTorch CUDA 版本匹配的 bitsandbytes/triton。"
    ) from e


from hulumed_qwen3.model import load_pretrained_model
from hulumed_qwen3.model.processor import HulumedProcessor
from hulumed_qwen3.mm_utils import load_images, get_model_name_from_path


# ==================== 1. 数据集类 ====================
class BUSIDataset(Dataset):
    """BUSI 乳腺癌数据集加载器"""

    def __init__(
            self,
            json_path: str,
            image_root: str,
            processor: HulumedProcessor,
            tokenizer,
            max_length: int = 2048,
    ):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"✅ 成功加载 {len(self.data)} 条训练数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 加载图像
        image_path = os.path.join(self.image_root, item["image"])
        try:
            image_data = load_images(image_path)
        except Exception as e:
            print(f"⚠️  图像加载失败: {image_path}, 错误: {e}")
            # 返回一个空白图像占位
            image_data = Image.new('RGB', (224, 224), (0, 0, 0))

        # 2. 构建对话格式（模仿推理时的格式）
        question = item["text"]
        answer = item["answer"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]

        # 3. 使用 processor 处理
        inputs = self.processor(
            images=[image_data],
            text=conversation,
            merge_size=1,  # 2D 图像
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_labels=True,  # 训练时需要 labels
        )

        # 4. 探针二分类标签：Normal=0, Benign/Malignant=1,2
        probe_label = item.get("label", 0)
        # if probe_label not in (0, 1):
        #     probe_label = 1 if probe_label >= 1 else 0
        inputs["probe_labels"] = torch.tensor(probe_label, dtype=torch.long)

        # 4. 将 batch 维度去掉（Dataset 返回单条数据）
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        return inputs


# ==================== 2. 数据整理器（Data Collator）====================
@dataclass
class DataCollatorForHulumed:
    """自定义数据整理器，处理变长序列和图像"""
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # 文本字段：动态补齐到当前 batch 的最大长度，避免不同样本长度导致 stack 失败
        if "input_ids" in features[0]:
            input_ids = [f["input_ids"].view(-1) for f in features]
            batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            )

        if "attention_mask" in features[0]:
            attention_masks = [f["attention_mask"].view(-1) for f in features]
            batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
        elif "input_ids" in batch:
            batch["attention_mask"] = (batch["input_ids"] != pad_token_id).long()

        if "labels" in features[0]:
            labels = [f["labels"].view(-1) for f in features]
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )

        # 图像字段：HuluMed 使用动态 resize，不同图像的 patch 数量不同（如 [1599,588] vs [1400,588]）
        # 不能 stack，需沿 dim=0 拼接；模型通过 grid_sizes 按样本切分
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "grid_sizes" in features[0]:
            gs_list = [f["grid_sizes"].squeeze().view(-1) for f in features]
            batch["grid_sizes"] = torch.stack(gs_list)  # [B, 3]
        if "merge_sizes" in features[0]:
            ms_list = [f["merge_sizes"].squeeze() for f in features]
            batch["merge_sizes"] = torch.stack(ms_list)  # [B]
        if "probe_labels" in features[0]:
            batch["probe_labels"] = torch.stack([f["probe_labels"] for f in features])

        return batch

        return batch


# ==================== 3. 训练配置 ====================
def get_lora_target_modules(model, target_suffixes: List[str], exclude_patterns: List[str] = None):
    """
    获取 LoRA 目标模块列表，支持排除指定模块（如视觉编码器）。

    因为 vision_encoder 里也有 q_proj/v_proj/k_proj 等，若直接用 target_suffixes，
    PEFT 会同时给视觉编码器和 LLM 加 LoRA。本函数通过 exclude_patterns 排除 vision_encoder。
    """
    exclude_patterns = exclude_patterns or ["vision_encoder"]

    module_names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in target_suffixes):
            if not any(exc in name for exc in exclude_patterns):
                module_names.append(name)

    return list(set(module_names))  # 去重


@dataclass
class ModelArguments:
    model_path: str = field(default="/ssd/common/LLMs/Hulu-Med-4B_finetune/normal/Hulu-Med-4B-merge-50-epoc")
    use_lora: bool = field(default=True)
    use_qlora: bool = field(default=False)  # 如果显存不够，改为 True
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    lora_exclude_patterns: str = field(default="vision_encoder")  # 排除的模块路径（逗号分隔），避免给视觉编码器加 LoRA
    train_mm_projector: bool = field(default=True)  # 是否同时训练映射层 mm_projector（视觉特征→LLM 的投影层）
    # 探针配置：vision_encoder_0, vision_encoder_5, projector, llm_0, llm_12 等；None 表示不使用探针
    # vision_encoder:0-26, llm:0-35, projector
    probe_position: Optional[str] = field(default=None)
    probe_type: str = field(default="mlp")
    probe_hidden_dim: Optional[int] = field(default=None)
    # 训练输出目录与运行名（供 run_train.sh 传入，避免多任务写同一路径）
    output_dir: str = field(default="/ssd/common/LLMs/Hulu-Med-4B_finetune/prober_4")
    run_name: str = field(default="hulu_med_busi_lora")


@dataclass
class DataArguments:
    train_json: str = field(default="/ssd/chenxi/Hulu-Med/BUSI/BUSI/train_busi_breast_cancer.json")
    image_root: str = field(default="/ssd/chenxi/Hulu-Med/BUSI/BUSI")
    max_length: int = field(default=2048)


# ==================== 4. 主训练函数 ====================
def train(model_args: ModelArguments, data_args: DataArguments):
    # --- 4.1 参数配置（model_args/data_args 由命令行解析，training_args 固定）---
    training_args = TrainingArguments(
        # 基础配置
        output_dir=model_args.output_dir,
        run_name=model_args.run_name,

        # 训练超参数
        num_train_epochs=50,  # 训练轮数
        per_device_train_batch_size=1,  # 每张卡的 batch size（根据显存调整）
        gradient_accumulation_steps=8,  # 梯度累积（相当于 batch_size=2*8=16）
        learning_rate=2e-4,  # 学习率（LoRA 建议 1e-4 到 5e-4）
        lr_scheduler_type="cosine",  # 学习率调度器（余弦退火）
        warmup_ratio=0.03,  # Warmup 比例（前 3% 的 step 做 warmup）
        weight_decay=0.01,  # 权重衰减
        max_grad_norm=1.0,  # 梯度裁剪

        # 优化器配置
        optim="adamw_torch",  # 优化器（或用 "paged_adamw_8bit" 节省显存）
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,

        # 日志和保存
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,  # 只保留最新的 1 个 checkpoint
        eval_strategy="no",  # 不做验证（如果有验证集改为 "steps"）

        # 数据类型和设备
        bf16=True,  # 使用 BF16（如果 GPU 支持）
        fp16=False,  # 如果不支持 BF16，改为 fp16=True
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # DeepSpeed 配置（如果要用 ZeRO-2/3，取消注释）
        # deepspeed="./ds_config_zero2.json",

        # 其他
        remove_unused_columns=False,  # 保留图像数据
        ddp_find_unused_parameters=False,  # 多卡训练优化
        report_to="tensorboard",  # 日志记录到 TensorBoard（或改为 "wandb"）
    )
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"📁 当前训练输出目录: {training_args.output_dir}")

    # --- 4.2 加载模型 ---
    print("🚀 正在加载预训练模型...")
    model_name = get_model_name_from_path(model_args.model_path)

    # 如果使用 QLoRA，需要 4bit 量化
    load_4bit = model_args.use_qlora

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_args.model_path,
        None,
        model_name,
        load_4bit=load_4bit,
        device_map='auto',  # 自动分配到多卡
        trust_remote_code=True,
        attn_implementation="sdpa",  # 或改为 "flash_attention_2"
    )

    processor = HulumedProcessor(image_processor, tokenizer)
    model.config.use_cache = False  # 训练时必须关闭 KV Cache
    model.config.use_token_compression = True

    # --- 4.2.1 配置探针（若启用）---
    if model_args.probe_position is not None:
        model.config.probe_position = model_args.probe_position
        model.config.probe_type = model_args.probe_type
        model.config.probe_hidden_dim = model_args.probe_hidden_dim
        meta_model = model.model if hasattr(model, 'model') else model
        if hasattr(meta_model, 'model'):
            meta_model = meta_model.model
        if getattr(meta_model, 'probe', None) is None:
            from hulumed_qwen3.model.probe import ProbeHead
            probe_input_dim = meta_model._get_probe_input_dim(meta_model.config, model_args.probe_position)
            meta_model.probe = ProbeHead(
                input_dim=probe_input_dim,
                num_classes=3,
                probe_type=model_args.probe_type,
                hidden_dim=model_args.probe_hidden_dim,
            )
            meta_model.probe.to(model.device)
        print(f"🔬 探针已启用: position={model_args.probe_position}, type={model_args.probe_type}")

    # --- 4.3 配置训练模式 (探针训练 vs LoRA 微调) ---
    if model_args.probe_position is not None:
        print("🔬 正在配置探针训练模式（冻结其余模块）...")
        # 1. 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 2. 只开启探针参数的梯度
        meta_model = model.model if hasattr(model, 'model') else model
        if hasattr(meta_model, 'model'):
            meta_model = meta_model.model
        if hasattr(meta_model, 'probe'):
            for param in meta_model.probe.parameters():
                param.requires_grad = True

        # 打印可训练参数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"   可训练参数: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.4f}%)")

    elif model_args.use_lora or model_args.use_qlora:
        print("🔧 正在配置 LoRA 微调...")

        if model_args.use_qlora:
            model = prepare_model_for_kbit_training(model)

        # 条件过滤：只对 LLM 加 LoRA，排除 vision_encoder（视觉编码器里也有 q_proj/v_proj 等）
        target_suffixes = [s.strip() for s in model_args.lora_target_modules.split(",")]
        exclude_patterns = [s.strip() for s in model_args.lora_exclude_patterns.split(",") if s.strip()]
        target_module_list = get_lora_target_modules(model, target_suffixes, exclude_patterns)
        print(f"   LoRA 目标模块数: {len(target_module_list)} (已排除: {exclude_patterns})")
        modules_to_save = ["model.mm_projector"] if model_args.train_mm_projector else None

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_module_list,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数量

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # --- 4.4 加载数据集 ---
    print("📊 正在加载训练数据...")
    train_dataset = BUSIDataset(
        json_path=data_args.train_json,
        image_root=data_args.image_root,
        processor=processor,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )

    data_collator = DataCollatorForHulumed(tokenizer=tokenizer)

    # --- 4.5 初始化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 4.6 开始训练 ---
    print("\n" + "=" * 60)
    print("🎯 开始训练 Hulu-Med LoRA 模型...")
    print("=" * 60 + "\n")

    trainer.train()

    # --- 4.7 保存最终模型 ---
    final_save_path = os.path.join(training_args.output_dir, "final_model")
    if model_args.probe_position is not None:
        # 1. 只保存探针权重
        print(f"💾 正在单独保存探针权重...")
        meta_model = model.model if hasattr(model, 'model') else model
        if hasattr(meta_model, 'model'):
            meta_model = meta_model.model

        if hasattr(meta_model, 'probe'):
            probe_weights = meta_model.probe.state_dict()
            # 根据位置和类型动态命名文件名
            probe_filename = f"probe_model_{model_args.probe_position}_{model_args.probe_type}.bin"
            save_path = os.path.join(training_args.output_dir, probe_filename)
            torch.save(probe_weights, save_path)
            print(f"✅ 探针权重已保存至: {save_path}")

        # 2. 如果你只想保存探针，甚至可以跳过下面的 trainer.save_model(final_save_path)
        # 但建议保留，因为它会保存训练配置和日志
        # trainer.save_model(final_save_path)
    else:
        # 正常保存 LoRA 模型
        trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    print(f"\n✅ 训练完成！模型已保存至: {final_save_path}")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args)
