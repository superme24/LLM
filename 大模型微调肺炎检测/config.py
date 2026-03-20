"""
================================================================================
项目配置文件 (config.py)
================================================================================
集中管理所有超参数和路径配置，方便统一修改。

数据集说明:
    本项目使用 COVID 胸部X光数据集，标注格式为二分类:
      - positive: 新冠肺炎阳性 (COVID-19 Positive)
      - negative: 新冠肺炎阴性 / 正常 (COVID-19 Negative / Normal)
    数据集结构:
      archive/
        train/       ← 训练集图片 (67,863 张)
        test/        ← 测试集图片 (8,482 张)
        val/         ← 验证集图片 (8,473 张)
        train.txt    ← 训练标注 (patient_id  filename  label  source)
        test.txt     ← 测试标注
        val.txt      ← 验证标注
"""

import os

# ======================== HuggingFace 镜像 ========================
# 如果未显式设置 HF_ENDPOINT，则默认使用国内镜像；
# 在海外云服务器上可通过环境变量改回 https://huggingface.co。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ======================== 路径配置 ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 原始数据集目录
DATA_ROOT = os.path.join(PROJECT_ROOT, "archive")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val")
TRAIN_ANNO = os.path.join(DATA_ROOT, "train.txt")
TEST_ANNO = os.path.join(DATA_ROOT, "test.txt")
VAL_ANNO = os.path.join(DATA_ROOT, "val.txt")

# 处理后的数据集 (JSON 格式) 保存路径
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
TRAIN_JSON = os.path.join(PROCESSED_DATA_DIR, "train.json")
VAL_JSON = os.path.join(PROCESSED_DATA_DIR, "val.json")
TEST_JSON = os.path.join(PROCESSED_DATA_DIR, "test.json")

# 模型输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LORA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "qwen_vl_lora")
MERGED_MODEL_DIR = os.path.join(OUTPUT_DIR, "qwen_vl_merged")
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "eval_results")

# ======================== 模型配置 ========================
# Qwen2-VL 基座模型
# Qwen2.5-VL 最小为 3B，加上 LoRA 训练开销超出 6GB，改用 Qwen2-VL-2B
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Medical LLaVA 模型 (用于对比实验)
MEDICAL_LLAVA_MODEL_NAME = "microsoft/llava-med-v1.5-mistral-7b"

# ======================== LoRA 超参数 ========================
LORA_R = 64              # LoRA 的秩 (rank)，决定低秩矩阵的维度
LORA_ALPHA = 128         # LoRA 缩放系数，通常设为 2*r
LORA_DROPOUT = 0.05      # Dropout 概率，防止过拟合
LORA_TARGET_MODULES = [   # 应用 LoRA 的目标模块
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# ======================== 训练超参数 ========================
NUM_EPOCHS = 3
BATCH_SIZE = 1                   # 6GB 显存建议 batch=1
GRADIENT_ACCUMULATION_STEPS = 16 # 等效 batch = 1 × 16 = 16
LEARNING_RATE = 2e-4             # 适当提高学习率，加快收敛
WARMUP_RATIO = 0.05              # 更长的 warmup，稳定早期训练
MAX_SEQ_LENGTH = 1024            # 6GB 显存下适当缩短
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100
FP16 = False
BF16 = True                     # RTX 3060 (Ampere) 支持 BF16
DATALOADER_NUM_WORKERS = 0 if os.name == "nt" else 4

# ======================== 数据集配置 ========================
IMAGE_SIZE = 448
SEED = 42
# 由于完整数据集 ~8.5 万张，VLM 微调时可采样子集加速训练
# 增大训练样本量以提升准确率和召回率（目标 ≥ 80%）
MAX_TRAIN_SAMPLES = 3000        # 训练采样数 (设为 None 使用全部)
MAX_VAL_SAMPLES = 500           # 验证采样数
MAX_TEST_SAMPLES = 500          # 测试采样数

# ======================== 类别标签映射 ========================
# 原始数据集的标签
LABEL_POSITIVE = "positive"     # COVID-19 阳性
LABEL_NEGATIVE = "negative"     # COVID-19 阴性 / 正常

# 中文标签 (用于构造 SFT 对话)
LABEL_MAP = {
    "positive": "新冠肺炎 (COVID-19 阳性)",
    "negative": "正常 (COVID-19 阴性)",
}

# 标签到数值的映射 (用于计算指标)
LABEL2ID = {"positive": 1, "negative": 0}
ID2LABEL = {0: "negative", 1: "positive"}

# ======================== Prompt 模板 ========================
# 用于 SFT 训练的系统提示
SYSTEM_PROMPT = (
    "你是一位经验丰富的放射科医生，擅长分析胸部X光片。"
    "请根据提供的胸部X光图像进行诊断分析。"
)

# 诊断报告提示 (训练用)
DIAGNOSIS_PROMPT = (
    "请仔细分析这张胸部X光片，判断该患者是否为新冠肺炎 (COVID-19) 阳性。\n"
    "请给出：\n"
    "1. 影像学表现\n"
    "2. 诊断结论（新冠肺炎阳性 / 正常）\n"
    "3. 诊断依据"
)

# 简单分类提示 (推理/对比用)
CLASSIFICATION_PROMPT = (
    "请判断这张胸部X光片是否为新冠肺炎 (COVID-19) 阳性。"
    "只需回答：新冠肺炎阳性 或 正常。"
)

# SFT 训练时阳性回复模板
POSITIVE_RESPONSE = (
    "根据影像学分析：\n"
    "1. 影像学表现：双肺可见多发磨玻璃影及实变影，分布以外周为主，"
    "伴有支气管充气征，符合病毒性肺炎的典型影像学表现。\n"
    "2. 诊断结论：新冠肺炎 (COVID-19) 阳性\n"
    "3. 诊断依据：双肺磨玻璃影呈外周分布，符合COVID-19肺炎的典型CT/X光表现。"
)

# SFT 训练时阴性回复模板
NEGATIVE_RESPONSE = (
    "根据影像学分析：\n"
    "1. 影像学表现：双肺纹理清晰，肺野透亮度正常，"
    "未见明显磨玻璃影、实变影或胸腔积液。\n"
    "2. 诊断结论：正常 (COVID-19 阴性)\n"
    "3. 诊断依据：胸部X光未见新冠肺炎相关的典型影像学改变。"
)
