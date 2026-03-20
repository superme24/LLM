"""
================================================================================
推理脚本 (inference.py)
================================================================================
功能:
    使用微调后的 Qwen2-VL + LoRA 模型对胸部X光片进行 COVID-19 诊断推理。
    支持单张图片推理和批量推理。

使用方法:
    # 单张图片推理
    python inference.py --image path/to/xray.png

    # 批量推理 (使用测试集)
    python inference.py --batch

    # 使用原始基座模型 (不加载 LoRA，用于对比)
    python inference.py --image path/to/xray.png --base-only
"""

import argparse
import json
import os
import sys

# config 必须在 transformers 之前导入，以设置 HF_ENDPOINT 镜像
import config

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_model(use_lora=True):
    """
    加载模型和处理器。

    参数:
        use_lora: 是否加载 LoRA 适配器 (False 则使用原始基座模型)

    返回:
        (model, processor) 元组
    """
    print("加载处理器...")
    processor = AutoProcessor.from_pretrained(
        config.QWEN_MODEL_NAME, trust_remote_code=True
    )

    print("加载基座模型...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_lora:
        if not os.path.isdir(config.LORA_OUTPUT_DIR):
            print(f"❌ LoRA 权重目录不存在: {config.LORA_OUTPUT_DIR}")
            print("  请先运行: python train_lora.py")
            sys.exit(1)
        print("加载 LoRA 适配器...")
        model = PeftModel.from_pretrained(model, config.LORA_OUTPUT_DIR)
        model = model.eval()
        print("✅ 已加载微调后的模型 (LoRA)")
    else:
        model = model.eval()
        print("✅ 使用原始基座模型 (未微调)")

    return model, processor


@torch.no_grad()
def predict_single(model, processor, image_path, prompt=None):
    """
    对单张胸部X光片进行推理。

    参数:
        model:      已加载的模型
        processor:  处理器
        image_path: 图像文件路径
        prompt:     推理提示 (默认使用 config.CLASSIFICATION_PROMPT)

    返回:
        str: 模型生成的诊断结果
    """
    if prompt is None:
        prompt = config.CLASSIFICATION_PROMPT

    # 构造消息
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(image_path).replace(os.sep, '/')}",
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # 应用聊天模板
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 处理视觉信息
    image_inputs, _ = process_vision_info(messages)

    # 编码输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )

    # 解码 (仅取生成的新 token)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def extract_prediction(response_text):
    """
    从模型回复中提取分类预测标签。

    优先匹配训练时使用的诊断报告关键词，支持中英文。
    采用加权评分：明确的诊断结论词权重高于泛化词汇。

    参数:
        response_text: 模型生成的文本

    返回:
        str: "positive" 或 "negative"
    """
    text = response_text.lower()

    # 高权重：明确的诊断结论词（来自训练时的 POSITIVE/NEGATIVE_RESPONSE 模板）
    strong_positive = ["covid-19 阳性", "新冠肺炎阳性", "新冠肺炎 (covid-19) 阳性",
                       "诊断结论：新冠肺炎", "诊断结论:新冠肺炎"]
    strong_negative = ["covid-19 阴性", "正常 (covid-19 阴性)", "covid-19) 阴性",
                       "诊断结论：正常", "诊断结论:正常",
                       "未见明显磨玻璃影", "未见新冠肺炎相关"]

    for kw in strong_positive:
        if kw in text:
            return "positive"
    for kw in strong_negative:
        if kw in text:
            return "negative"

    # 普通权重关键词
    positive_keywords = ["阳性", "新冠肺炎", "covid", "positive",
                         "磨玻璃影", "实变影", "病毒性肺炎"]
    negative_keywords = ["阴性", "正常", "normal", "negative", "未见", "healthy"]

    pos_score = sum(1 for kw in positive_keywords if kw in text)
    neg_score = sum(1 for kw in negative_keywords if kw in text)

    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    else:
        # 无法确定时默认 negative（减少误报，提升特异度）
        return "negative"


def batch_predict(model, processor, test_json_path, output_path=None):
    """
    批量推理测试集。

    使用与训练时相同的 DIAGNOSIS_PROMPT，以保证分布一致性，
    再从详细诊断报告中提取分类标签。

    参数:
        model:           已加载的模型
        processor:       处理器
        test_json_path:  测试集 JSON 文件路径
        output_path:     结果保存路径

    返回:
        list[dict]: 推理结果列表
    """
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\n开始批量推理 ({len(test_data)} 条)...\n")

    results = []
    for i, sample in enumerate(test_data):
        # 从 messages 中提取图像路径
        image_url = None
        for msg in sample["messages"]:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        image_url = item["image"]
                        break

        if not image_url:
            continue

        # 提取本地路径
        image_path = image_url.replace("file://", "").replace("/", os.sep)

        # 从 assistant 回复中提取真实标签（使用 extract_prediction 保持一致）
        true_label = None
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                true_label = extract_prediction(msg["content"])

        # 使用与训练一致的 DIAGNOSIS_PROMPT 推理，再提取标签
        response = predict_single(
            model, processor, image_path, prompt=config.DIAGNOSIS_PROMPT
        )
        pred_label = extract_prediction(response)

        results.append({
            "image": os.path.basename(image_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "response": response,
        })

        # 进度显示
        correct = "✅" if pred_label == true_label else "❌"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_data)}] {correct} "
                  f"真实: {true_label}, 预测: {pred_label}")

    # 计算准确率
    correct_count = sum(1 for r in results if r["true_label"] == r["pred_label"])
    accuracy = correct_count / len(results) if results else 0
    print(f"\n批量推理完成: 准确率 = {correct_count}/{len(results)} = {accuracy:.2%}")

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="COVID-19 胸部X光诊断推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--batch", action="store_true", help="批量推理测试集")
    parser.add_argument("--base-only", action="store_true",
                        help="仅使用基座模型 (不加载 LoRA)")
    parser.add_argument("--diagnosis", action="store_true",
                        help="生成详细诊断报告 (默认为简单分类)")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.print_help()
        print("\n请指定 --image 或 --batch 参数")
        sys.exit(1)

    # 加载模型
    model, processor = load_model(use_lora=not args.base_only)

    if args.image:
        # 单张推理
        if not os.path.isfile(args.image):
            print(f"❌ 图片不存在: {args.image}")
            sys.exit(1)

        prompt = config.DIAGNOSIS_PROMPT if args.diagnosis else config.CLASSIFICATION_PROMPT
        print(f"\n分析图像: {args.image}")
        print("-" * 40)
        response = predict_single(model, processor, args.image, prompt)
        print(f"\n诊断结果:\n{response}")
        pred = extract_prediction(response)
        print(f"\n分类: {config.LABEL_MAP[pred]}")

    if args.batch:
        # 批量推理
        if not os.path.isfile(config.TEST_JSON):
            print("❌ 测试数据不存在，请先运行: python prepare_data.py")
            sys.exit(1)

        output_path = os.path.join(config.EVAL_OUTPUT_DIR, "qwen_vl_predictions.json")
        batch_predict(model, processor, config.TEST_JSON, output_path)


if __name__ == "__main__":
    main()
