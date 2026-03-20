# 大模型 LoRA 微调 — COVID-19 胸部X光诊断

基于 **Qwen2-VL-2B-Instruct** 视觉语言模型，使用 **LoRA（Low-Rank Adaptation）** 技术对 COVID-19 胸部X光数据集进行监督微调（SFT），实现胸部X光片的新冠肺炎二分类诊断。

目标：**准确率（Accuracy）≥ 80%，召回率（Recall）≥ 80%**。

---

## 项目结构

```
大模型微调肺炎检测/
├── archive/                  # 原始数据集 (需自行放置)
│   ├── train/                # 训练集图片 (~67,863 张)
│   ├── test/                 # 测试集图片 (~8,482 张)
│   ├── val/                  # 验证集图片 (~8,473 张)
│   ├── train.txt             # 训练标注文件
│   ├── test.txt              # 测试标注文件
│   └── val.txt               # 验证标注文件
├── processed_data/           # 转换后的 SFT 数据 (自动生成)
│   ├── train.json
│   ├── val.json
│   └── test.json
├── output/                   # 模型输出 (自动生成)
│   ├── qwen_vl_lora/         # LoRA 权重
│   ├── qwen_vl_merged/       # 合并后的完整模型 (可选)
│   └── eval_results/         # 评估结果与可视化
├── config.py                 # 统一配置文件
├── prepare_data.py           # 数据预处理脚本
├── train_lora.py             # LoRA 微调训练脚本
├── inference.py              # 推理脚本
├── evaluate.py               # 评估脚本
├── compare_llava.py          # Medical LLaVA 对比实验脚本
└── requirements.txt          # Python 依赖
```

---

## 数据集说明

标注文件每行格式（空格分隔）：

```
patient_id  image_filename  label  source
```

| 字段 | 说明 |
|------|------|
| `patient_id` | 患者 ID |
| `image_filename` | 图像文件名（含扩展名） |
| `label` | `positive`（COVID-19 阳性）或 `negative`（阴性/正常） |
| `source` | 数据来源 |

---

## 技术架构

### LoRA 原理

对于预训练权重矩阵 **W ∈ ℝ^(d×k)**，LoRA 将更新拆解为低秩分解：

```
W' = W + ΔW = W + B·A
```

其中 **B ∈ ℝ^(d×r)**，**A ∈ ℝ^(r×k)**，**r ≪ min(d, k)**。

- 只训练 B 和 A，参数量从 d×k 降到 r×(d+k)
- 本项目 r=64，可训练参数仅约为全量微调的 ~1%

### 模型选择

| 组件 | 选择 | 理由 |
|------|------|------|
| 基座模型 | Qwen2-VL-2B-Instruct | 2B 参数适合 6GB 显存；支持图文多模态 |
| 微调方法 | LoRA (rank=64, alpha=128) | 低显存、快速收敛 |
| 精度 | BF16 | Ampere 架构 GPU 原生支持，速度快 |
| 训练框架 | TRL SFTTrainer | 原生支持对话格式 SFT |

---

## 环境配置

### 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| 显存 | 6 GB | 12 GB+ |
| 内存 | 16 GB | 32 GB |
| 存储 | 20 GB | 50 GB |

### 安装依赖

```bash
pip install -r requirements.txt
```

> **国内用户**：默认已配置 `https://hf-mirror.com` 镜像，无需额外设置。
> 若在海外服务器，可设置环境变量：`export HF_ENDPOINT=https://huggingface.co`

---

## 快速开始

### 第一步：准备数据

```bash
python prepare_data.py
```

输出示例：
```
处理训练集 (train)...
  标注文件共 67863 条记录
  有效样本: 67863, 图像缺失: 0
    新冠肺炎 (COVID-19 阳性): 34283 (50.5%)
    正常 (COVID-19 阴性): 33580 (49.5%)
  采样后: 2000 条 (positive: 1010, negative: 990)
  ✅ 已保存到: processed_data/train.json
```

### 第二步：LoRA 微调训练

```bash
python train_lora.py
```

训练过程输出：
```
[1/5] 加载 Qwen2-VL 模型和处理器...
[2/5] 配置 LoRA 适配器...
  可训练比例: ~1.2%
[3/5] 加载训练和验证数据...
[4/5] 配置训练参数...
[5/5] 开始训练...
  Step 10/375 | Loss: 1.234
  ...
训练完成！LoRA 权重保存于: output/qwen_vl_lora/
```

支持断点续训（自动检测 checkpoint）。

**（可选）合并 LoRA 权重到基座模型（推理更快）：**

```bash
python train_lora.py --merge
```

### 第三步：批量推理

```bash
python inference.py --batch
```

### 第四步：评估性能

```bash
# 基础评估
python evaluate.py

# 评估 + 生成可视化图表
python evaluate.py --plot

# 对比 Medical LLaVA（需先运行 compare_llava.py）
python evaluate.py --compare
```

### 第五步（可选）：与 Medical LLaVA 对比

```bash
python compare_llava.py
```

### 单张图片推理

```bash
# 简单分类（只输出阳性/阴性）
python inference.py --image path/to/xray.png

# 详细诊断报告
python inference.py --image path/to/xray.png --diagnosis

# 使用原始基座模型（不加载 LoRA）
python inference.py --image path/to/xray.png --base-only
```

---

## 评估指标说明

| 指标 | 公式 | 含义 |
|------|------|------|
| 准确率 (Accuracy) | (TP+TN)/(TP+TN+FP+FN) | 整体分类正确率 |
| 精确率 (Precision) | TP/(TP+FP) | 预测阳性中真正阳性比例 |
| **召回率 (Recall)** | TP/(TP+FN) | 所有阳性中被正确识别比例 |
| F1 分数 | 2×P×R/(P+R) | 精确率与召回率的调和平均 |
| 特异度 (Specificity) | TN/(TN+FP) | 所有阴性中被正确识别比例 |

> 医学诊断中，**召回率（灵敏度）** 尤为重要——漏诊阳性（FN）的危害远大于误诊（FP）。

**目标：Accuracy ≥ 80%，Recall ≥ 80%**

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LORA_R` | 64 | LoRA 秩（rank），越大容量越强但显存越多 |
| `LORA_ALPHA` | 128 | LoRA 缩放系数，通常 = 2×r |
| `LORA_DROPOUT` | 0.05 | 防止过拟合 |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `MAX_TRAIN_SAMPLES` | 2000 | 训练采样数（None=全量） |
| `MAX_SEQ_LENGTH` | 1024 | 最大序列长度 |

---

## 常见问题

**Q: 显存不足 (OOM)**

调小以下参数（`config.py`）：
```python
MAX_SEQ_LENGTH = 512      # 缩短序列长度
LORA_R = 32               # 降低 LoRA 秩
MAX_TRAIN_SAMPLES = 500   # 减少训练样本
```

**Q: 下载模型很慢**

已默认启用国内镜像 `https://hf-mirror.com`，如仍较慢可手动预下载：
```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
```

**Q: 准确率 / 召回率未达标**

- 增大 `MAX_TRAIN_SAMPLES`（如 5000）
- 增加 `NUM_EPOCHS`（如 5）
- 调低 `LEARNING_RATE`（如 5e-5）
- 检查正负样本是否均衡（prepare_data.py 已做分层采样）

---

## 项目亮点

1. **完整 Pipeline**：数据预处理 → LoRA 微调 → 推理 → 评估 → 对比实验，端到端可复现
2. **低资源友好**：6GB 显存可训练，LoRA 可训练参数仅 ~1%
3. **分层采样**：自动保持正负样本均衡，保证召回率
4. **断点续训**：自动检测 checkpoint，支持意外中断后继续训练
5. **可视化评估**：混淆矩阵、指标对比柱状图，一键生成
6. **对比实验**：与 Medical LLaVA 零样本结果对比，量化 SFT 微调的价值
