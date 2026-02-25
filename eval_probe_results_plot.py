import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class ProbeEvalResult:
    file_name: str
    file_path: str
    probe_position: str
    probe_type: str
    total: int
    valid: int
    correct: int
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 probe 结果并画准确率图。")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/ssd/chenxi/Hulu-Med/x_code/infer_result",
        help="包含 probe 结果 json 的目录",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default=None,
        help="输出图片路径，默认保存到 input-dir/probe_accuracy.png",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="输出汇总 json 路径，默认保存到 input-dir/probe_accuracy_summary.json",
    )
    return parser.parse_args()


def discover_result_files(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"目录不存在: {input_dir}")

    files = []
    for name in os.listdir(input_dir):
        if not name.endswith(".json"):
            continue
        # 兼容 probe_model_ 前缀和误写的 robe_model_ 前缀
        if name.startswith("probe_model_") or name.startswith("robe_model_"):
            files.append(os.path.join(input_dir, name))
    return sorted(files)


def safe_load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_from_filename(file_name: str) -> Tuple[str, str]:
    # 例: probe_model_llm_10_mlp_results.json
    pattern = re.compile(r"^(?:p?robe_model)_(.+)_(mlp|linear)(?:_results)?\.json$")
    m = pattern.match(file_name)
    if not m:
        return "unknown", "unknown"
    return m.group(1), m.group(2)


def extract_probe_info(data, file_name: str) -> Tuple[str, str]:
    probe_position, probe_type = parse_from_filename(file_name)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        item0 = data[0]
        active_probe = item0.get("active_probe", {}) if isinstance(item0.get("active_probe"), dict) else {}
        probe_output = item0.get("probe_output", {}) if isinstance(item0.get("probe_output"), dict) else {}
        probe_position = active_probe.get("probe_position") or probe_output.get("probe_position") or probe_position
        probe_type = active_probe.get("probe_type") or probe_output.get("probe_type") or probe_type
    return str(probe_position), str(probe_type)


def evaluate_one_file(file_path: str) -> ProbeEvalResult:
    file_name = os.path.basename(file_path)
    data = safe_load_json(file_path)
    if not isinstance(data, list):
        raise ValueError(f"文件不是 list 格式: {file_name}")

    probe_position, probe_type = extract_probe_info(data, file_name)

    total = len(data)
    valid = 0
    correct = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        true_label = item.get("label", None)
        probe_output = item.get("probe_output", {})
        pred_label = probe_output.get("probe_pred_id", None) if isinstance(probe_output, dict) else None

        if true_label is None or pred_label is None:
            continue

        try:
            true_label = int(true_label)
            pred_label = int(pred_label)
        except Exception:
            continue

        valid += 1
        if true_label == pred_label:
            correct += 1

    accuracy = (correct / valid) if valid > 0 else 0.0
    return ProbeEvalResult(
        file_name=file_name,
        file_path=file_path,
        probe_position=probe_position,
        probe_type=probe_type,
        total=total,
        valid=valid,
        correct=correct,
        accuracy=accuracy,
    )


def sort_key(res: ProbeEvalResult):
    pos = res.probe_position
    stage_rank = 3
    layer_idx = 10**9

    if pos.startswith("vision_encoder_"):
        stage_rank = 0
        try:
            layer_idx = int(pos.split("_")[-1])
        except Exception:
            layer_idx = 10**9
    elif pos == "projector" or pos.startswith("projector"):
        stage_rank = 1
        layer_idx = 0
    elif pos.startswith("llm_"):
        stage_rank = 2
        try:
            layer_idx = int(pos.split("_")[-1])
        except Exception:
            layer_idx = 10**9

    type_rank = 0 if res.probe_type == "linear" else 1
    return (stage_rank, layer_idx, type_rank, res.file_name)


def build_display_labels(results: List[ProbeEvalResult]) -> List[str]:
    labels = [r.probe_position for r in results]
    duplicated = {x for x in labels if labels.count(x) > 1}
    out = []
    for r in results:
        if r.probe_position in duplicated:
            out.append(f"{r.probe_position}({r.probe_type})")
        else:
            out.append(r.probe_position)
    return out


def plot_results(results: List[ProbeEvalResult], output_fig: str):
    labels = build_display_labels(results)
    values = [r.accuracy * 100 for r in results]

    width = max(12, len(labels) * 0.6)
    plt.figure(figsize=(width, 6))
    bars = plt.bar(range(len(labels)), values)

    plt.xticks(range(len(labels)), labels, rotation=55, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Probe Position")
    plt.title("Probe Accuracy by Position")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_fig, dpi=220)
    plt.close()


def save_summary(results: List[ProbeEvalResult], output_json: str):
    payload = [
        {
            "file_name": r.file_name,
            "probe_position": r.probe_position,
            "probe_type": r.probe_type,
            "total_items": r.total,
            "valid_items": r.valid,
            "correct_items": r.correct,
            "accuracy": r.accuracy,
        }
        for r in results
    ]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    output_fig = args.output_fig or os.path.join(args.input_dir, "probe_accuracy.png")
    output_json = args.output_json or os.path.join(args.input_dir, "probe_accuracy_summary.json")

    files = discover_result_files(args.input_dir)
    if not files:
        print(f"未找到匹配文件（probe_model_*.json / robe_model_*.json）: {args.input_dir}")
        return

    results = []
    for fp in files:
        try:
            results.append(evaluate_one_file(fp))
        except Exception as e:
            print(f"[跳过] {os.path.basename(fp)}: {e}")

    if not results:
        print("没有可统计的结果文件。")
        return

    results.sort(key=sort_key)
    save_summary(results, output_json)
    plot_results(results, output_fig)

    print("\n=== Probe 准确率统计 ===")
    for r in results:
        print(
            f"{r.probe_position:>20s} ({r.probe_type:<6s}) | "
            f"acc={r.accuracy*100:6.2f}% | valid={r.valid:4d}/{r.total:4d} | file={r.file_name}"
        )
    print(f"\n汇总文件已保存: {output_json}")
    print(f"图像已保存: {output_fig}")


if __name__ == "__main__":
    main()
