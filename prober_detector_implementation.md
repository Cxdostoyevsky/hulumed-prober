探针（Probe）检测器实现计划

1. 架构与数据流概览

flowchart TB
    subgraph VisionPath [视觉路径]
        VE[Vision Encoder]
        VE_L0[Layer 0]
        VE_L1[Layer 1]
        VE_LN[Layer N]
    end
    subgraph Connector [连接器]
        MM[mm_projector]
    end
    subgraph LLMPath [LLM 路径]
        EMB[embed_tokens]
        LLM_L0[LLM Layer 0]
        LLM_LN[LLM Layer N]
    end
    VE --> VE_L0 --> VE_L1 --> VE_LN --> MM --> EMB --> LLM_L0 --> LLM_LN
    VE_L0 -.->|probe_vision_0| P1[Probe]
    MM -.->|probe_projector| P2[Probe]
    LLM_L0 -.->|probe_llm_0| P3[Probe]

维度对应关系：





视觉编码器各层输出：mm_hidden_size（如 768）



projector 输出：config.hidden_size（LLM 维度，如 3584）



LLM 各层输出：config.hidden_size（3584）



2. 探针类型选择

推荐：2 层 MLP（Linear → GELU → Linear → 2 维 logits）





比单层 Linear 更能拟合非线性决策边界



参数量适中，不易过拟合



二分类任务常用结构



3. 实现方案

3.1 配置与参数

在 hulumed_qwen2.py 的 HulumedQwen2Config / HulumedQwen3Config 中新增：

probe_position: Optional[str] = None   # "vision_encoder_0", "vision_encoder_5", "projector", "llm_0", "llm_12" 等
probe_type: str = "mlp"               # "mlp" 或 "linear"
probe_hidden_dim: Optional[int] = None  # MLP 中间层维度，默认 input_dim // 2

probe_position 格式约定：





vision_encoder_{i}：视觉编码器第 i 层输出后（i ∈ [0, num_hidden_layers-1]，如 12 层则 0~11）



projector：mm_projector 输出后



llm_{i}：LLM 第 i 层输出后（i ∈ [0, num_hidden_layers-1]）

3.2 探针模块定义

新建 src/hulumed_qwen3/model/probe.py：

class ProbeHead(nn.Module):
    """二分类探针：支持 Linear 或 2 层 MLP"""
    def __init__(self, input_dim: int, num_classes: int = 2, probe_type: str = "mlp", hidden_dim: Optional[int] = None):
        ...

3.3 模型侧改动

文件： hulumed_arch.py





在 HulumedMetaModel.__init__ 中，若 config.probe_position 非空，则根据位置计算 probe_input_dim 并构建 self.probe。



在 HulumedMetaForCausalLM 中新增 encode_images_for_probe（或扩展现有 encode_images）：





vision_encoder_X：修改视觉编码器 forward，支持在指定层后返回中间特征（需在 modeling_hulumed_encoder.py 的 HulumedVisionTransformerEncoder.forward 中支持 stop_at_layer 参数）。



projector：调用 vision_encoder + mm_projector，返回投影后的特征。



llm_X：在 prepare_inputs_labels_for_multimodal 中构造 inputs_embeds，再在 HulumedQwen2ForCausalLM.forward 中只跑 LLM 前 X 层。

文件： hulumed_qwen2.py





在 HulumedQwen2ForCausalLM.forward 中：





若 probe_labels 存在且 config.probe_position 非空：





根据 probe_position 调用对应路径，得到 probe_features。



按 batched_num_patches / grid_sizes 做 per-image mean pooling 得到 [B, D]。



probe_logits = self.model.probe(probe_features)。



probe_loss = F.cross_entropy(probe_logits, probe_labels)。



返回 CausalLMOutputWithPast(loss=probe_loss, ...)，不执行后续 LLM forward。



否则：保持原有 forward 逻辑。

3.4 特征聚合（per-image pooling）

视觉编码器 / projector 输出形状为 [N_patches_total, D]，需按样本切分并做 mean pooling：





使用 grid_sizes、merge_sizes、batched_num_patches 计算每个样本的 patch 数量。



按样本切分后对每个样本做 mean(dim=0)，得到 [B, D]。

3.5 视觉编码器中间层支持

文件： modeling_hulumed_encoder.py





在 HulumedVisionTransformerEncoder.forward 中增加可选参数 stop_at_layer: Optional[int] = None。



若指定，则只执行 layers[0:stop_at_layer+1] 并返回，用于探针在某一层后截断。

3.6 数据与训练脚本

文件： train_busi_lora.py





在 BUSIDataset.__getitem__ 中增加 probe_labels：从 item["label"] 读取，若为三分类可映射为二分类（如 Normal=0, Benign+Malignant=1）。



在 DataCollatorForHulumed 中增加 probe_labels 的 batch 拼接。



在 ModelArguments 中增加 probe_position 等参数。



加载模型后，将 probe_position 写入 model.config，并确保探针模块被正确初始化。



4. 前向判断逻辑（伪代码）

def forward(..., probe_labels=None, **kwargs):
    if probe_labels is not None and self.config.probe_position is not None:
        # 探针模式：数据流到探针处截断
        pos = parse_probe_position(self.config.probe_position)  # (stage, layer_idx)
        if pos.stage == "vision_encoder":
            features = encode_vision_up_to_layer(pos.layer_idx, pixel_values, grid_sizes, merge_sizes)
        elif pos.stage == "projector":
            features = encode_images(pixel_values, grid_sizes, merge_sizes)  # vision + projector
        else:  # llm
            inputs_embeds = prepare_inputs_labels_for_multimodal(...)
            features = run_llm_layers_0_to_N(inputs_embeds, pos.layer_idx)
        features = per_image_mean_pool(features, batched_num_patches)
        logits = self.model.probe(features)
        loss = F.cross_entropy(logits, probe_labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    # 正常 VLM 模式
    ...



5. 涉及文件清单







文件



改动





hulumed_qwen2.py



Config 增加 probe 参数；Forward 增加探针分支





hulumed_arch.py



构建 probe 模块；实现 encode_for_probe / 特征获取逻辑





probe.py



新建 ProbeHead





modeling_hulumed_encoder.py



Vision encoder 支持 stop_at_layer





train_busi_lora.py



Dataset / Collator 增加 probe_labels；ModelArguments 增加 probe 参数



6. 注意事项





探针与 LoRA：探针单独训练时，可冻结主模型，只训练 model.probe；或与 LoRA 联合训练。



三分类转二分类：若需 Normal vs Abnormal，可将 label in [1,2] 映射为 1。



config 持久化：若从 checkpoint 加载，需在 config.json 中保存 probe_position、probe_type 等，以便正确恢复探针结构。

