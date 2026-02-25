import sys
import os
import re
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch
from hulumed_qwen3.model import load_pretrained_model
from hulumed_qwen3.mm_utils import (
    load_images,
    load_video,
    get_model_name_from_path
)
from hulumed_qwen3.model.processor import HulumedProcessor
import json 

PROBE_LABEL_MAP = {0: "Normal", 1: "Benign", 2: "Malignant"}


def discover_probe_checkpoints(probe_dir):
    pattern = re.compile(r"^probe_model_(.+)_(mlp|linear)\.bin$")
    probes = []
    if not probe_dir or not os.path.isdir(probe_dir):
        return probes

    for file_name in sorted(os.listdir(probe_dir)):
        match = pattern.match(file_name)
        if not match:
            continue
        probe_position = match.group(1)
        probe_type = match.group(2)
        probes.append(
            {
                "name": file_name,
                "path": os.path.join(probe_dir, file_name),
                "probe_position": probe_position,
                "probe_type": probe_type,
            }
        )
    return probes


def run_inference():
    # --- 1. 配置路径 ---
    model_path = "/ssd/common/LLMs/Hulu-Med-4B"
    probe_dir = "/ssd/common/LLMs/Hulu-Med-4B_finetune/prober"
    json_path = "/Users/chenxi/Desktop/Hulu-Med/test_busi_breast_cancer.json"
    image_root = "/ssd/chenxi/Hulu-Med/BUSI/BUSI"
    
    model_name = get_model_name_from_path(model_path)
    device = "cuda:1"
    print(f"正在从本地加载模型: {model_name}...")

    # --- 2. 加载模型和处理器 ---
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    processor = HulumedProcessor(image_processor, tokenizer)
    model.config.use_token_compression = True
    model = model.to(device)
    probe_specs = discover_probe_checkpoints(probe_dir)
    print(f"已发现探针权重数: {len(probe_specs)}")
    for spec in probe_specs:
        print(f"  - {spec['name']} => position={spec['probe_position']}, type={spec['probe_type']}")

    # --- 3. 加载 JSON 数据并开始推理 ---
    if not os.path.exists(json_path):
        print(f"错误: 找不到 JSON 文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"\n开始处理下游任务，共 {len(data_list)} 条数据...")

    if not probe_specs:
        print("未发现探针权重，将仅执行基础生成推理。")
        results = []
        for i, item in enumerate(data_list):
            image_relative_path = item.get("image")
            full_image_path = os.path.join(image_root, image_relative_path)
            question = item.get("text", "Please analyze this image.")
            print(f"[{i+1}/{len(data_list)}] 正在处理: {image_relative_path}")
            if not os.path.exists(full_image_path):
                print(f"跳过: 找不到图像 {full_image_path}")
                continue
            image_data = load_images(full_image_path)
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": question}],
                }
            ]
            response, probe_output = process_and_generate(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                visual_data=image_data,
                conversation=conversation,
                modal='image',
                device=device,
                probe_spec=None,
            )
            out_item = dict(item)
            out_item["model_output"] = response
            out_item["probe_output"] = probe_output
            results.append(out_item)
        output_save_path = "inference_results_busi.json"
        with open(output_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n推理完成！结果已保存至: {output_save_path}")
        return
    else:
        # 外层按探针循环：加载一个探针 -> 跑完整测试集 -> 保存结果 -> 下一个探针
        for probe_idx, spec in enumerate(probe_specs):
            print(
                f"\n===== [{probe_idx + 1}/{len(probe_specs)}] 当前探针: {spec['name']} "
                f"(position={spec['probe_position']}, type={spec['probe_type']}) ====="
            )
            model.load_probe_weights(
                probe_ckpt_path=spec["path"],
                probe_position=spec["probe_position"],
                probe_type=spec["probe_type"],
            )
            results = []
            for i, item in enumerate(data_list):
                image_relative_path = item.get("image")
                full_image_path = os.path.join(image_root, image_relative_path)
                question = item.get("text", "Please analyze this image.")
                print(f"[probe {probe_idx+1}/{len(probe_specs)} | sample {i+1}/{len(data_list)}] {image_relative_path}")
                if not os.path.exists(full_image_path):
                    print(f"跳过: 找不到图像 {full_image_path}")
                    continue
                image_data = load_images(full_image_path)
                conversation = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": question}],
                    }
                ]
                response, probe_output = process_and_generate(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    visual_data=image_data,
                    conversation=conversation,
                    modal='image',
                    device=device,
                    probe_spec=spec,
                )
                out_item = dict(item)
                out_item["active_probe"] = {
                    "probe_name": spec["name"],
                    "probe_position": spec["probe_position"],
                    "probe_type": spec["probe_type"],
                }
                out_item["model_output"] = response
                out_item["probe_output"] = probe_output
                results.append(out_item)

            output_name = os.path.splitext(spec["name"])[0] + "_results.json"
            output_save_path = os.path.join(".", output_name)
            with open(output_save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"探针 {spec['name']} 推理完成，结果已保存至: {output_save_path}")

# 修改一下 process_and_generate 让它返回字符串，方便保存
def process_and_generate(model, processor, tokenizer, visual_data, conversation, modal, device, probe_spec=None):
    """通用的处理和生成函数"""
    inputs = processor(
        images=[visual_data] if modal != "text" else None,
        text=conversation,
        merge_size=2 if modal == "video" else 1,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    probe_output = None
    if probe_spec is not None:
        with torch.inference_mode():
            probe_result = model.predict_probe(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                pixel_values=inputs.get("pixel_values"),
                grid_sizes=inputs.get("grid_sizes"),
                merge_sizes=inputs.get("merge_sizes"),
                modals=[modal],
            )
        pred_id = int(probe_result["preds"][0].item())
        probs = probe_result["probs"][0].detach().float().cpu().tolist()
        probe_output = {
            "probe_name": probe_spec["name"],
            "probe_position": probe_spec["probe_position"],
            "probe_type": probe_spec["probe_type"],
            "probe_pred_id": pred_id,
            "probe_pred_label": PROBE_LABEL_MAP.get(pred_id, str(pred_id)),
            "probe_probs": probs,
        }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            modals=[modal],
            temperature=0.6,
            max_new_tokens=128, # 下游分类任务不需要太长，128足够
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"模型回答: {outputs}")
    return outputs, probe_output


if __name__ == "__main__":
    run_inference()