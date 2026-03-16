"""
================================================================================
LoRA 微调训练脚本 (train_lora.py)
================================================================================
功能:
    使用 LoRA (Low-Rank Adaptation) 对 Qwen2-VL-2B-Instruct 进行
    监督微调 (SFT)，使其具备 COVID-19 胸部X光诊断能力。

核心技术:
    1. LoRA — 仅训练低秩分解矩阵，大幅降低显存需求
    2. SFT  — 监督微调，使用对话格式的标注数据
    3. BF16 — 混合精度训练，加速计算

使用方法:
    python train_lora.py

依赖:
    - transformers, peft, trl, qwen-vl-utils, accelerate
    - 需要先运行 prepare_data.py 生成训练数据
"""

import json
import os
import sys

# config 必须在 transformers 之前导入，以设置 HF_ENDPOINT 镜像
import config

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# def load_sft_dataset(json_path):
#     """
#     加载 SFT 格式的 JSON 数据集。

#     参数:
#         json_path: JSON 文件路径

#     返回:
#         list[dict]: 对话数据列表
#     """
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     print(f"  加载 {len(data)} 条数据: {json_path}")
#     return data
def load_sft_dataset(json_path):
    """
    加载 SFT 格式的 JSON 数据集（支持标准JSON数组和每行一个JSON对象两种格式）
    """
    print(f"📂 加载数据集: {json_path}")
    data = []

    try:
        # 第一步：尝试以标准 JSON 数组格式加载（优先）
        with open(json_path, 'r', encoding='utf-8') as f:
            # 先读取全部内容并去除首尾空白，避免 BOM/换行干扰
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                data = json.loads(content)
                print(f"  ✅ 检测为标准JSON数组格式，共 {len(data)} 条")
                return data
    except json.JSONDecodeError as e:
        print(f"  ⚠️ 标准JSON数组解析失败，尝试逐行加载: {e}")

    # 第二步：逐行加载（每行一个 JSON 对象）
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            print(f"  📄 检测为多行JSON格式，逐行加载...")
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ 第 {line_num + 1} 行解析失败: {e}")
                    print(f"     问题行内容: {line[:100]}...")
                    continue
        print(f"  ✅ 成功加载 {len(data)} 条数据")
    except Exception as e:
        raise ValueError(f"数据集加载失败: {json_path}") from e

    return data


def create_lora_config():
    """
    创建 LoRA 配置。

    LoRA 原理简述:
        对于预训练权重矩阵 W ∈ R^{d×k}，LoRA 将更新拆解为:
            W' = W + ΔW = W + B·A
        其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
        这样只需训练 B 和 A，参数量从 d×k 降到 r×(d+k)。

    返回:
        LoraConfig: PEFT LoRA 配置对象
    """
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return lora_config


def main():
    """主训练流程。"""
    print("=" * 60)
    print("Qwen2-VL LoRA 微调 — COVID-19 胸部X光诊断")
    print("=" * 60)

    # ============================================================
    # 第1步: 检查训练数据是否存在
    # ============================================================
    if not os.path.isfile(config.TRAIN_JSON):
        print("❌ 训练数据不存在，请先运行: python prepare_data.py")
        sys.exit(1)

    # ============================================================
    # 第2步: 加载模型和处理器
    # ============================================================
    print("\n[1/5] 加载 Qwen2-VL 模型和处理器...")

    # 加载处理器 (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(
        config.QWEN_MODEL_NAME,
        trust_remote_code=True,
    )

    # 加载预训练模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 开启梯度检查点以节省显存
    model.enable_input_require_grads()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ============================================================
    # 第3步: 添加 LoRA 适配器
    # ============================================================
    print("\n[2/5] 配置 LoRA 适配器...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params / 1e6:.1f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.1f}M")
    print(f"  可训练比例: {trainable_params / total_params * 100:.2f}%")
    model.print_trainable_parameters()

    # ============================================================
    # 第4步: 加载数据集
    # ============================================================
    print("\n[3/5] 加载训练和验证数据...")
    train_data = load_sft_dataset(config.TRAIN_JSON)
    val_data = load_sft_dataset(config.VAL_JSON) if os.path.isfile(config.VAL_JSON) else None

    # ============================================================
    # 第5步: 配置训练参数
    # ============================================================
    print("\n[4/5] 配置训练参数...")
    os.makedirs(config.LORA_OUTPUT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=config.LORA_OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS if val_data else None,
        eval_strategy="steps" if val_data else "no",
        save_total_limit=3,
        bf16=config.BF16,
        fp16=config.FP16,
        max_length=config.MAX_SEQ_LENGTH,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        remove_unused_columns=False,
        seed=config.SEED,
        report_to="none",
        gradient_checkpointing=True,
        # SFT 相关
        dataset_text_field=None,  # 使用 messages 格式
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
    )

    print(f"  输出目录: {config.LORA_OUTPUT_DIR}")
    print(f"  训练轮数: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE} × {config.GRADIENT_ACCUMULATION_STEPS} = "
          f"{config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  精度: {'BF16' if config.BF16 else 'FP16' if config.FP16 else 'FP32'}")

    # ============================================================
    # 第6步: 定义数据整理函数 (collate_fn)
    # ============================================================
    def collate_fn(examples):
        """
        数据整理函数: 将一个 batch 的对话数据转换为模型输入。

        对于 Qwen-VL 模型，需要特殊处理:
          1. 使用 processor.apply_chat_template 将对话转为文本
          2. 使用 processor 同时处理文本和图像
          3. 构造 labels (仅对 assistant 回复计算损失)
        """
        texts = []
        images_list = []

        for example in examples:
            messages = example["messages"]
            # 将对话转为文本
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # 提取图像路径
            images = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            images.append(item["image"])
            images_list.append(images)

        # 使用 processor 处理文本+图像
        from qwen_vl_utils import process_vision_info

        # 为每个样本提取视觉信息
        all_images = []
        for msgs_example in examples:
            img_inputs, _ = process_vision_info(msgs_example["messages"])
            if img_inputs:
                all_images.extend(img_inputs)

        batch = processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )

        # 构造 labels: 将非 assistant 部分的 token 设为 -100 (不计算损失)
        labels = batch["input_ids"].clone()
        # 找到 assistant 回复的边界并只保留这部分的 labels
        # 简化处理: 使用 padding token 掩码
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    # ============================================================
    # 第7步: 创建训练器并开始训练
    # ============================================================
    print("\n[5/5] 开始训练...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # 开始训练 (支持从 checkpoint 断点续训)
    last_checkpoint = None
    if os.path.isdir(config.LORA_OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(config.LORA_OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(config.LORA_OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
            print(f"  发现 checkpoint，断点续训: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 保存最终模型
    print("\n保存 LoRA 适配器权重...")
    trainer.save_model(config.LORA_OUTPUT_DIR)
    processor.save_pretrained(config.LORA_OUTPUT_DIR)

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}")
    print(f"  LoRA 权重保存于: {config.LORA_OUTPUT_DIR}")
    print(f"  训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  训练步数: {metrics.get('train_steps', 'N/A')}")


def merge_lora():
    """
    将 LoRA 适配器权重合并到基础模型中，保存完整模型。
    (可选步骤，合并后推理更快但占用更多磁盘空间)

    使用方法:
        python train_lora.py --merge
    """
    print("=" * 60)
    print("合并 LoRA 权重到基础模型")
    print("=" * 60)

    from peft import PeftModel

    # 加载基础模型
    print("加载基础模型...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA 适配器
    print("加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, config.LORA_OUTPUT_DIR)

    # 合并权重
    print("合并权重...")
    model = model.merge_and_unload()

    # 保存合并后的模型
    os.makedirs(config.MERGED_MODEL_DIR, exist_ok=True)
    print(f"保存到: {config.MERGED_MODEL_DIR}")
    model.save_pretrained(config.MERGED_MODEL_DIR)

    # 保存 processor
    processor = AutoProcessor.from_pretrained(
        config.QWEN_MODEL_NAME, trust_remote_code=True
    )
    processor.save_pretrained(config.MERGED_MODEL_DIR)

    print("✅ 合并完成！")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge_lora()
    else:
        main()
