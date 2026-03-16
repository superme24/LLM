"""
================================================================================
数据准备脚本 (prepare_data.py)
================================================================================
功能:
    1. 读取 archive/ 下的 train.txt / val.txt / test.txt 标注文件
    2. 将原始标注转换为 Qwen-VL SFT 训练所需的多轮对话 JSON 格式
    3. 支持采样子集以加速 VLM 微调 (完整数据集 ~8.5万张)
    4. 输出统计信息

标注文件格式 (每行用空格分隔):
    patient_id  image_filename  label(positive/negative)  source

输出 JSON 格式 (Qwen-VL SFT 格式):
    [
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": [
                    {"type": "image", "image": "file:///path/to/image.png"},
                    {"type": "text", "text": "诊断提示..."}
                ]},
                {"role": "assistant", "content": "诊断结果..."}
            ]
        },
        ...
    ]
"""

import json
import os
import random
from collections import Counter

import config


def parse_annotation_file(anno_path):
    """
    解析标注文件，返回 (filename, label) 列表。

    参数:
        anno_path: 标注文件路径 (如 train.txt)

    返回:
        list of dict: [{"filename": "xxx.png", "label": "positive"}, ...]
    """
    samples = []
    with open(anno_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            # 格式: patient_id  filename  label  source
            filename = parts[1]
            label = parts[2]
            if label not in (config.LABEL_POSITIVE, config.LABEL_NEGATIVE):
                print(f"⚠ 跳过未知标签: {label} (文件: {filename})")
                continue
            samples.append({"filename": filename, "label": label})
    return samples


def build_sft_conversation(image_path, label):
    """
    构建单条 Qwen-VL SFT 训练数据 (多轮对话格式)。

    参数:
        image_path: 图像的绝对路径
        label: 标签 ("positive" 或 "negative")

    返回:
        dict: 包含 messages 列表的字典
    """
    # 根据标签选择回复模板
    if label == config.LABEL_POSITIVE:
        response = config.POSITIVE_RESPONSE
    else:
        response = config.NEGATIVE_RESPONSE

    conversation = {
        "messages": [
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path.replace(os.sep, '/')}",
                    },
                    {
                        "type": "text",
                        "text": config.DIAGNOSIS_PROMPT,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]
    }
    return conversation


def process_split(anno_path, img_dir, output_json, max_samples=None, split_name="train"):
    """
    处理一个数据集划分 (train/val/test)。

    参数:
        anno_path:   标注文件路径
        img_dir:     图像目录路径
        output_json: 输出 JSON 文件路径
        max_samples: 最大采样数 (None 表示使用全部)
        split_name:  划分名称 (用于打印)
    """
    print(f"\n{'='*60}")
    print(f"处理 {split_name} 数据集...")
    print(f"{'='*60}")

    # 1. 解析标注
    samples = parse_annotation_file(anno_path)
    print(f"  标注文件共 {len(samples)} 条记录")

    # 2. 验证图像文件是否存在
    valid_samples = []
    missing_count = 0
    for s in samples:
        img_path = os.path.join(img_dir, s["filename"])
        if os.path.isfile(img_path):
            s["image_path"] = img_path
            valid_samples.append(s)
        else:
            missing_count += 1
    print(f"  有效样本: {len(valid_samples)}, 图像缺失: {missing_count}")

    # 3. 统计标签分布
    label_counts = Counter(s["label"] for s in valid_samples)
    for label, count in sorted(label_counts.items()):
        cn_label = config.LABEL_MAP.get(label, label)
        print(f"    {cn_label}: {count} ({count/len(valid_samples)*100:.1f}%)")

    # 4. 采样 (如果指定了 max_samples)
    if max_samples and len(valid_samples) > max_samples:
        random.seed(config.SEED)
        # 分层采样，保持正负样本比例
        pos_samples = [s for s in valid_samples if s["label"] == config.LABEL_POSITIVE]
        neg_samples = [s for s in valid_samples if s["label"] == config.LABEL_NEGATIVE]
        pos_ratio = len(pos_samples) / len(valid_samples)
        n_pos = int(max_samples * pos_ratio)
        n_neg = max_samples - n_pos
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        valid_samples = pos_samples[:n_pos] + neg_samples[:n_neg]
        random.shuffle(valid_samples)
        print(f"  采样后: {len(valid_samples)} 条 (positive: {n_pos}, negative: {n_neg})")

    # 5. 构建 SFT 对话数据
    sft_data = []
    for s in valid_samples:
        conv = build_sft_conversation(s["image_path"], s["label"])
        sft_data.append(conv)

    # 6. 保存 JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    # with open(output_json, "w", encoding="utf-8") as f:
    #     json.dump(sft_data, f, ensure_ascii=False, indent=2)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)  # 关键：ensure_ascii=False

    print(f"  ✅ 已保存到: {output_json}")
    print(f"  共 {len(sft_data)} 条 SFT 训练数据")

    return sft_data


def main():
    """主函数: 处理 train / val / test 三个数据集划分。"""
    print("=" * 60)
    print("COVID-19 胸部X光数据集 → Qwen-VL SFT 格式转换")
    print("=" * 60)

    # 处理训练集
    train_data = process_split(
        anno_path=config.TRAIN_ANNO,
        img_dir=config.TRAIN_IMG_DIR,
        output_json=config.TRAIN_JSON,
        max_samples=config.MAX_TRAIN_SAMPLES,
        split_name="训练集 (train)",
    )

    # 处理验证集
    val_data = process_split(
        anno_path=config.VAL_ANNO,
        img_dir=config.VAL_IMG_DIR,
        output_json=config.VAL_JSON,
        max_samples=config.MAX_VAL_SAMPLES,
        split_name="验证集 (val)",
    )

    # 处理测试集
    test_data = process_split(
        anno_path=config.TEST_ANNO,
        img_dir=config.TEST_IMG_DIR,
        output_json=config.TEST_JSON,
        max_samples=config.MAX_TEST_SAMPLES,
        split_name="测试集 (test)",
    )

    # 总结
    print(f"\n{'='*60}")
    print("数据准备完成！")
    print(f"{'='*60}")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    print(f"  输出目录: {config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
