"""
================================================================================
Medical LLaVA 对比实验脚本 (compare_llava.py)
================================================================================
功能:
    使用 Medical LLaVA (llava-med-v1.5-mistral-7b) 对同一测试集进行 COVID-19
    诊断推理，与微调后的 Qwen2-VL 进行性能对比。

对比维度:
    1. 分类准确率 (Accuracy)
    2. 精确率/召回率/F1 (Precision/Recall/F1)
    3. 推理速度 (Throughput)
    4. 生成回复的质量 (定性分析)

使用方法:
    python compare_llava.py

注意:
    Medical LLaVA 是基于 LLaVA 架构的医学多模态模型，未经过本数据集微调。
    对比的目的是展示 SFT 微调的效果 vs 通用医学模型的零样本能力。
"""

import json
import os
import sys
import time

# config 必须在 transformers 之前导入，以设置 HF_ENDPOINT 镜像
import config

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor


def load_medical_llava():
    """
    加载 Medical LLaVA 模型。

    Medical LLaVA 基于 LLaVA-1.5 架构 + Mistral-7B 语言模型，
    使用 PMC-15M 医学图文数据集预训练，具有一定的医学图像理解能力。

    返回:
        (model, tokenizer, image_processor) 元组
    """
    print("加载 Medical LLaVA 模型...")
    print(f"  模型: {config.MEDICAL_LLAVA_MODEL_NAME}")

    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(config.MEDICAL_LLAVA_MODEL_NAME)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            config.MEDICAL_LLAVA_MODEL_NAME,
            None,  # model_base
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("✅ Medical LLaVA 加载成功 (使用 llava 库)")
        return model, tokenizer, image_processor, "llava_native"

    except ImportError:
        # 如果没安装 llava 库，使用 transformers 直接加载
        print("  llava 库未安装，尝试使用 transformers 加载...")
        try:
            from transformers import LlavaForConditionalGeneration

            processor = AutoProcessor.from_pretrained(
                config.MEDICAL_LLAVA_MODEL_NAME,
                trust_remote_code=True,
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                config.MEDICAL_LLAVA_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = model.eval()
            print("✅ Medical LLaVA 加载成功 (使用 transformers)")
            return model, processor, None, "transformers"

        except Exception as e:
            print(f"❌ 无法加载 Medical LLaVA: {e}")
            print("  请安装: pip install llava 或确保模型可用")
            return None, None, None, None


@torch.no_grad()
def predict_with_llava_transformers(model, processor, image_path, prompt):
    """使用 transformers 版 LLaVA 进行推理。"""
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )

    # 解码
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    # 提取 assistant 回复部分
    if "[/INST]" in generated_text:
        generated_text = generated_text.split("[/INST]")[-1].strip()
    return generated_text


@torch.no_grad()
def predict_with_llava_native(model, tokenizer, image_processor, image_path, prompt):
    """使用原生 llava 库进行推理。"""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates["mistral_instruct"].copy()
    input_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return output_text


def extract_prediction(response_text):
    """从模型回复中提取分类标签。"""
    text = response_text.lower()
    positive_keywords = ["阳性", "新冠", "covid", "positive", "磨玻璃", "实变",
                         "coronavirus", "infected", "pneumonia"]
    negative_keywords = ["阴性", "正常", "normal", "negative", "healthy",
                         "no evidence", "clear", "未见"]

    pos_score = sum(1 for kw in positive_keywords if kw in text)
    neg_score = sum(1 for kw in negative_keywords if kw in text)

    return "positive" if pos_score > neg_score else "negative"


def run_comparison(test_json_path, output_dir):
    """
    运行对比实验: Medical LLaVA vs Qwen2-VL (LoRA微调)。

    流程:
        1. 加载 Medical LLaVA
        2. 对测试集进行推理
        3. 保存结果
        4. 与 Qwen-VL 结果合并对比
    """

    # ============================================================
    # 1. 加载数据
    # ============================================================
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条\n")

    # ============================================================
    # 2. 加载 Medical LLaVA
    # ============================================================
    model, tokenizer_or_processor, image_processor, backend = load_medical_llava()
    if model is None:
        print("跳过 Medical LLaVA 对比 (模型加载失败)")
        return

    # ============================================================
    # 3. 推理
    # ============================================================
    # 使用英文提示 (Medical LLaVA 主要训练于英文数据)
    english_prompt = (
        "Analyze this chest X-ray image. Is the patient COVID-19 positive or negative? "
        "Provide your assessment including imaging findings and conclusion."
    )

    results = []
    start_time = time.time()

    for i, sample in enumerate(test_data):
        # 提取图像路径
        image_url = None
        for msg in sample["messages"]:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        image_url = item["image"]
                        break
        if not image_url:
            continue

        image_path = image_url.replace("file:///", "").replace("/", os.sep)

        # 提取真实标签
        true_label = None
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                true_label = "positive" if "阳性" in msg["content"] else "negative"

        # 推理
        try:
            if backend == "transformers":
                response = predict_with_llava_transformers(
                    model, tokenizer_or_processor, image_path, english_prompt
                )
            else:
                response = predict_with_llava_native(
                    model, tokenizer_or_processor, image_processor,
                    image_path, english_prompt
                )
        except Exception as e:
            response = f"Error: {str(e)}"

        pred_label = extract_prediction(response)

        results.append({
            "image": os.path.basename(image_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "response": response,
        })

        correct = "✅" if pred_label == true_label else "❌"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_data)}] {correct} "
                  f"真实: {true_label}, 预测: {pred_label}")

    elapsed = time.time() - start_time

    # ============================================================
    # 4. 统计和保存
    # ============================================================
    correct_count = sum(1 for r in results if r["true_label"] == r["pred_label"])
    accuracy = correct_count / len(results) if results else 0

    print(f"\nMedical LLaVA 推理结果:")
    print(f"  准确率: {correct_count}/{len(results)} = {accuracy:.2%}")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  平均: {elapsed/len(results):.2f}s/张" if results else "")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    llava_result_path = os.path.join(output_dir, "llava_predictions.json")
    with open(llava_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  结果保存于: {llava_result_path}")

    # ============================================================
    # 5. 对比汇总
    # ============================================================
    qwen_result_path = os.path.join(output_dir, "qwen_vl_predictions.json")
    if os.path.isfile(qwen_result_path):
        with open(qwen_result_path, "r", encoding="utf-8") as f:
            qwen_results = json.load(f)

        qwen_correct = sum(1 for r in qwen_results
                          if r["true_label"] == r["pred_label"])
        qwen_acc = qwen_correct / len(qwen_results) if qwen_results else 0

        print(f"\n{'='*60}")
        print("模型对比汇总")
        print(f"{'='*60}")
        print(f"{'模型':<30} {'准确率':<15} {'样本数':<10}")
        print(f"{'-'*55}")
        print(f"{'Qwen2-VL + LoRA (微调)':<28} {qwen_acc:<15.2%} {len(qwen_results):<10}")
        print(f"{'Medical LLaVA (零样本)':<28} {accuracy:<15.2%} {len(results):<10}")
        print(f"\n提升: {(qwen_acc - accuracy):.2%}")
    else:
        print(f"\n提示: 运行 'python inference.py --batch' 生成 Qwen-VL 推理结果后可进行对比")


def main():
    print("=" * 60)
    print("Medical LLaVA 对比实验")
    print("=" * 60)

    if not os.path.isfile(config.TEST_JSON):
        print("❌ 测试数据不存在，请先运行: python prepare_data.py")
        sys.exit(1)

    run_comparison(config.TEST_JSON, config.EVAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
