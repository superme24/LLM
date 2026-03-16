"""
================================================================================
评估脚本 (evaluate.py)
================================================================================
功能:
    对模型推理结果进行全面的性能评估，生成评估报告。

评估指标:
    1. 准确率 (Accuracy)
    2. 精确率 (Precision) — 预测为阳性中真正阳性的比例
    3. 召回率 (Recall / Sensitivity) — 所有阳性中被正确识别的比例
    4. F1 分数 — 精确率和召回率的调和平均
    5. 特异度 (Specificity) — 所有阴性中被正确识别的比例
    6. 混淆矩阵 (Confusion Matrix)
    7. ROC 曲线 / AUC (如可用)

使用方法:
    # 评估 Qwen-VL 结果
    python evaluate.py

    # 评估并生成可视化
    python evaluate.py --plot

    # 对比评估两个模型
    python evaluate.py --compare
"""

import argparse
import json
import os
import sys
from collections import Counter

import config


def compute_metrics(results):
    """
    计算二分类评估指标。

    参数:
        results: list[dict], 每个字典包含 "true_label" 和 "pred_label"

    返回:
        dict: 包含各项指标的字典

    指标说明:
        TP (True Positive):  真阳性 — 实际阳性且预测阳性
        TN (True Negative):  真阴性 — 实际阴性且预测阴性
        FP (False Positive): 假阳性 — 实际阴性但预测阳性 (误报)
        FN (False Negative): 假阴性 — 实际阳性但预测阴性 (漏报)
    """
    tp = fp = tn = fn = 0
    for r in results:
        true = r["true_label"]
        pred = r["pred_label"]
        if true == "positive" and pred == "positive":
            tp += 1
        elif true == "negative" and pred == "negative":
            tn += 1
        elif true == "negative" and pred == "positive":
            fp += 1
        elif true == "positive" and pred == "negative":
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }
    return metrics


def print_metrics(metrics, model_name="模型"):
    """打印评估指标。"""
    print(f"\n{'='*50}")
    print(f"  {model_name} 评估结果")
    print(f"{'='*50}")
    print(f"  样本总数: {metrics['total']}")
    print(f"  准确率 (Accuracy):    {metrics['accuracy']:.4f}  ({metrics['accuracy']:.2%})")
    print(f"  精确率 (Precision):   {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):      {metrics['recall']:.4f}")
    print(f"  F1 分数:              {metrics['f1']:.4f}")
    print(f"  特异度 (Specificity): {metrics['specificity']:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"  {'':>20}  预测阳性  预测阴性")
    print(f"  {'实际阳性':>18}   {metrics['tp']:>5}    {metrics['fn']:>5}")
    print(f"  {'实际阴性':>18}   {metrics['fp']:>5}    {metrics['tn']:>5}")


def print_comparison(qwen_metrics, llava_metrics):
    """打印两个模型的对比。"""
    print(f"\n{'='*60}")
    print("  模型对比: Qwen2-VL (LoRA微调) vs Medical LLaVA (零样本)")
    print(f"{'='*60}")

    metric_names = [
        ("accuracy",    "准确率 (Accuracy)"),
        ("precision",   "精确率 (Precision)"),
        ("recall",      "召回率 (Recall)"),
        ("f1",          "F1 分数"),
        ("specificity", "特异度 (Specificity)"),
    ]

    print(f"\n  {'指标':<22} {'Qwen-VL (LoRA)':>15} {'Medical LLaVA':>15} {'差值':>10}")
    print(f"  {'-'*62}")
    for key, name in metric_names:
        q = qwen_metrics[key]
        l = llava_metrics[key]
        diff = q - l
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {name:<20} {q:>14.4f} {l:>14.4f} {arrow}{abs(diff):>8.4f}")

    print(f"\n  结论:")
    if qwen_metrics["f1"] > llava_metrics["f1"]:
        improvement = (qwen_metrics["f1"] - llava_metrics["f1"]) / llava_metrics["f1"] * 100 \
            if llava_metrics["f1"] > 0 else float("inf")
        print(f"  → 微调后的 Qwen2-VL 在 F1 指标上优于 Medical LLaVA，"
              f"提升约 {improvement:.1f}%")
        print(f"  → 这说明针对特定任务的 SFT 微调能显著提升模型在该任务上的表现，")
        print(f"    即使基座模型并非专门的医学模型。")
    else:
        print(f"  → Medical LLaVA 在该任务上表现优于微调后的 Qwen2-VL，")
        print(f"    可能需要更多训练数据或更优的超参数。")


def generate_plots(qwen_results, llava_results, output_dir):
    """
    生成可视化图表。

    包括:
      1. 混淆矩阵热力图
      2. 指标对比柱状图
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("⚠ matplotlib 未安装，跳过图表生成")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. 混淆矩阵热力图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (results, name) in zip(axes, [
        (qwen_results, "Qwen2-VL + LoRA"),
        (llava_results, "Medical LLaVA"),
    ]):
        if results is None:
            ax.set_visible(False)
            continue
        m = compute_metrics(results)
        cm = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["阳性", "阴性"])
        ax.set_yticklabels(["阳性", "阴性"])
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_title(f"{name}\n准确率: {m['accuracy']:.2%}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵已保存: {cm_path}")

    # ---- 2. 指标对比柱状图 ----
    if llava_results:
        qm = compute_metrics(qwen_results)
        lm = compute_metrics(llava_results)

        metric_names = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
        metric_keys = ["accuracy", "precision", "recall", "f1", "specificity"]

        qwen_vals = [qm[k] for k in metric_keys]
        llava_vals = [lm[k] for k in metric_keys]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, qwen_vals, width,
                       label="Qwen2-VL + LoRA", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, llava_vals, width,
                       label="Medical LLaVA", color="#DD8452")

        ax.set_ylabel("分数")
        ax.set_title("模型性能对比")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # 在柱状图上标注数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        comp_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(comp_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  对比图已保存: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="COVID-19 诊断模型评估")
    parser.add_argument("--compare", action="store_true", help="对比两个模型")
    parser.add_argument("--plot", action="store_true", help="生成可视化图表")
    args = parser.parse_args()

    # 加载 Qwen-VL 推理结果
    qwen_path = os.path.join(config.EVAL_OUTPUT_DIR, "qwen_vl_predictions.json")
    llava_path = os.path.join(config.EVAL_OUTPUT_DIR, "llava_predictions.json")

    qwen_results = None
    llava_results = None

    if os.path.isfile(qwen_path):
        with open(qwen_path, "r", encoding="utf-8") as f:
            qwen_results = json.load(f)
        qwen_metrics = compute_metrics(qwen_results)
        print_metrics(qwen_metrics, "Qwen2-VL + LoRA (微调)")
    else:
        print(f"⚠ Qwen-VL 推理结果不存在: {qwen_path}")
        print("  请先运行: python inference.py --batch")

    if args.compare or os.path.isfile(llava_path):
        if os.path.isfile(llava_path):
            with open(llava_path, "r", encoding="utf-8") as f:
                llava_results = json.load(f)
            llava_metrics = compute_metrics(llava_results)
            print_metrics(llava_metrics, "Medical LLaVA (零样本)")
        else:
            print(f"\n⚠ Medical LLaVA 推理结果不存在: {llava_path}")
            print("  请先运行: python compare_llava.py")

    # 对比
    if qwen_results and llava_results:
        print_comparison(qwen_metrics, llava_metrics)

    # 生成图表
    if args.plot and qwen_results:
        print("\n生成可视化图表...")
        generate_plots(qwen_results, llava_results, config.EVAL_OUTPUT_DIR)

    # 保存评估报告
    if qwen_results:
        report = {
            "qwen_vl_metrics": compute_metrics(qwen_results),
        }
        if llava_results:
            report["llava_metrics"] = compute_metrics(llava_results)

        os.makedirs(config.EVAL_OUTPUT_DIR, exist_ok=True)
        report_path = os.path.join(config.EVAL_OUTPUT_DIR, "eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
