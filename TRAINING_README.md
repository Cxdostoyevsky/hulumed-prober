# 🏥 Hulu-Med BUSI 乳腺癌微调指南

## 📋 项目说明
本指南帮助你使用 **LoRA/QLoRA** 在 **BUSI 乳腺癌数据集**上微调 Hulu-Med-4B 模型。

---

## 🚀 快速开始

### 1️⃣ 环境准备
确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### 2️⃣ 数据准备
确认以下文件路径正确：
- **训练数据**: `/Users/chenxi/Desktop/Hulu-Med/train_busi_breast_cancer.json`
- **图像根目录**: `/ssd/chenxi/Hulu-Med/BUSI/BUSI`
- **预训练模型**: `/ssd/common/LLMs/Hulu-Med-4B`

如果路径不同，请修改 `train_busi_lora.py` 中的以下位置：
```python
model_path: str = field(default="/ssd/common/LLMs/Hulu-Med-4B")
train_json: str = field(default="/Users/chenxi/Desktop/Hulu-Med/train_busi_breast_cancer.json")
image_root: str = field(default="/ssd/chenxi/Hulu-Med/BUSI/BUSI")
```

### 3️⃣ 开始训练

#### 方式 A：单卡训练
```bash
CUDA_VISIBLE_DEVICES=1 python train_busi_lora.py
```

#### 方式 B：多卡训练（推荐）
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_busi_lora.py
```

#### 方式 C：使用 DeepSpeed ZeRO-2（多卡 + 节省显存）
```bash
# 1. 修改 train_busi_lora.py 中的 TrainingArguments，取消注释：
# deepspeed="./ds_config_zero2.json",

# 2. 运行训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_busi_lora.py
```

---

## ⚙️ 超参数说明

### 核心训练参数（在 `train_busi_lora.py` 中修改）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_epochs` | `3` | 训练轮数 |
| `per_device_train_batch_size` | `2` | 每张卡的 batch size |
| `gradient_accumulation_steps` | `8` | 梯度累积步数（相当于 batch=16） |
| `learning_rate` | `2e-4` | 学习率（LoRA 推荐 1e-4~5e-4） |
| `warmup_ratio` | `0.03` | Warmup 比例（前 3% 的 step） |
| `weight_decay` | `0.01` | 权重衰减（L2 正则化） |
| `lr_scheduler_type` | `"cosine"` | 学习率调度器（余弦退火） |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_lora` | `True` | 是否使用 LoRA |
| `use_qlora` | `False` | 是否使用 QLoRA（4bit 量化，节省显存） |
| `lora_r` | `64` | LoRA 秩（越大参数越多，建议 32~128） |
| `lora_alpha` | `128` | LoRA 缩放因子（一般设为 2*r） |
| `lora_dropout` | `0.05` | LoRA Dropout 比例 |
| `lora_target_modules` | `q_proj,v_proj,...` | 要应用 LoRA 的模块 |
| `lora_exclude_patterns` | `vision_encoder` | 排除的模块（避免给视觉编码器加 LoRA） |
| `train_mm_projector` | `False` | 是否同时训练映射层 mm_projector（视觉→LLM 投影层） |

---

## 💾 显存优化建议

### 显存不足？试试这些方法：

#### 1. **启用 QLoRA（4bit 量化）**
修改 `ModelArguments` 中的参数：
```python
use_qlora: bool = field(default=True)  # 改为 True
```

#### 2. **减小 batch size**
```python
per_device_train_batch_size=1,  # 改为 1
gradient_accumulation_steps=16,  # 增加梯度累积
```

#### 3. **减小 LoRA 秩**
```python
lora_r: int = field(default=32)  # 从 64 改为 32
```

#### 4. **启用梯度检查点**（在 `TrainingArguments` 中添加）
```python
gradient_checkpointing=True,
```

#### 5. **使用 8bit 优化器**
```python
optim="paged_adamw_8bit",
```

---

## 📊 训练监控

### TensorBoard 实时查看
```bash
tensorboard --logdir=/ssd/chenxi/Hulu-Med/checkpoints/busi_lora/runs
```

### 或使用 WandB（需在 `TrainingArguments` 中设置）
```python
report_to="wandb",
```
然后运行：
```bash
wandb login  # 首次使用需要登录
python train_busi_lora.py
```

---

## 📁 输出文件

训练完成后，模型会保存在：
```
/ssd/chenxi/Hulu-Med/checkpoints/busi_lora/
├── checkpoint-100/       # 中间检查点
├── checkpoint-200/
├── final_model/          # 最终模型
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer配置文件
└── runs/                 # TensorBoard 日志
```

---

## 🧪 训练后推理

修改 `inference_hulumed_qwen3.py` 加载你的 LoRA 模型：

```python
# 在 load_pretrained_model 之前添加：
from peft import PeftModel

# 加载基础模型
tokenizer, model, image_processor, context_len = load_pretrained_model(...)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(
    model, 
    "/ssd/chenxi/Hulu-Med/checkpoints/busi_lora/final_model"
)
model = model.merge_and_unload()  # 合并权重
```

---

## ❓ 常见问题

### Q1: `CUDA out of memory` 错误
**A**: 参考上面的"显存优化建议"，依次尝试：QLoRA → 减小 batch size → 减小 LoRA 秩。

### Q2: 训练速度很慢？
**A**: 
1. 检查是否使用了 `bf16=True`（或 `fp16=True`）
2. 如果有多张卡，使用 `torchrun` 多卡训练
3. 启用 DeepSpeed ZeRO-2

### Q3: 如何调整训练轮数？
**A**: 修改 `num_train_epochs`，或使用 `max_steps` 替代：
```python
max_steps=1000,  # 训练 1000 步后停止
num_train_epochs=None,  # 设为 None
```

### Q4: 想要保存更多 checkpoint？
**A**: 修改 `save_total_limit`：
```python
save_total_limit=5,  # 保留最新的 5 个 checkpoint
```

---

## 📞 需要帮助？
- 查看训练日志：`cat /ssd/chenxi/Hulu-Med/checkpoints/busi_lora/trainer_state.json`
- 检查模型配置：`cat /ssd/common/LLMs/Hulu-Med-4B/config.json`

祝训练顺利！🎉
