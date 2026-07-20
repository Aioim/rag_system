# 模型微调 & 蒸馏 — 操作手册

> 适用版本：`src/model/__init__.py` v1.1.0+
> 最后更新：2026-07-12

---

## 1. 概述

`src/model/finetune/` 模块在模型下载管理的基础上，提供三种模型的 **LoRA 微调** 能力和 LLM 的 **黑盒知识蒸馏** 能力。

```
┌─────────────────────────────────────────────────────┐
│                  模型生命周期                         │
│                                                      │
│  HuggingFace Hub ──→ 下载基座模型 ──→ 微调/蒸馏       │
│       ↑                    ↑              │          │
│  ModelDownloader      ModelManager    BaseTrainer    │
│                                         │   │   │    │
│                              Embedding  Reranker LLM │
│                                                     │
│  输出: local_models/finetuned/{name}/                      │
│    ├── adapter_config.json                           │
│    ├── adapter_model.safetensors                     │
│    └── metadata.yaml                                 │
└─────────────────────────────────────────────────────┘
```

| 能力 | Embedding | Reranker | LLM |
|------|:---:|:---:|:---:|
| 基座模型 | `BAAI/bge-large-zh-v1.5` | `BAAI/bge-reranker-v2-m3` | `Qwen/Qwen3-0.6B` |
| 微调方式 | LoRA | LoRA | LoRA (SFT) |
| 损失函数 | MultipleNegativesRankingLoss | CrossEntropy (二分类) | CrossEntropy |
| 蒸馏 | — | — | 黑盒蒸馏（云端 API → 本地小模型） |
| 显存需求 (LoRA) | ~4 GB | ~4 GB | ~6 GB (0.6B) |

---

## 2. 前置条件

### 2.1 安装依赖

```bash
pip install rag-service[finetune]
```

这等价于：

```bash
pip install peft>=0.12.0 datasets>=2.19.0 accelerate>=0.30.0
```

此外还需要训练目标对应的依赖：

```bash
# Embedding / Reranker 微调需要
pip install sentence-transformers>=3.3

# LLM 蒸馏需要（调用云端教师 API）
pip install openai
```

### 2.2 硬件要求

| 模型 | 最低显存 | 推荐显存 |
|------|---------|---------|
| Embedding (1024-dim) | 4 GB | 8 GB |
| Reranker (CrossEncoder) | 4 GB | 8 GB |
| LLM (0.6B Qwen) | 6 GB | 12 GB |
| CPU 训练 | 可行但极慢 | 不推荐 |

### 2.3 API 密钥

```bash
# .env 文件
HUGGINGFACE_TOKEN=hf_xxx          # 下载 BGE/Qwen 基座模型
ANTHROPIC_API_KEY=sk-ant-xxx      # LLM 蒸馏的教师 API（可选，仅蒸馏需要）
```

---

## 3. 配置

### 3.1 YAML 配置（`config/{env}.yaml`，如 `config/dev.yaml`）

```yaml
finetune:
  output_dir: local_models/finetuned       # LoRA 适配器输出目录
  device: auto                       # auto | cuda | cpu
  data_dir: data/finetune            # 训练数据默认目录

  training:
    epochs: 3
    learning_rate: 2.0e-4
    batch_size: 8
    warmup_ratio: 0.1
    max_seq_length: 512
    gradient_accumulation_steps: 4
    eval_steps: 100
    save_steps: 500
    logging_steps: 50

  lora:
    r: 8                              # LoRA rank（越大越强但越慢）
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: null              # null=自动推断

  distillation:
    temperature: 2.0                  # 预留（当前黑盒蒸馏未使用）
    alpha: 0.5                        # 硬标签权重（0=纯蒸馏, 1=纯SFT）
```

### 3.2 环境变量覆盖

遵循项目统一的 `双下划线` 规则：

```bash
export FINETUNE__TRAINING__EPOCHS=5
export FINETUNE__TRAINING__BATCH_SIZE=4
export FINETUNE__DEVICE=cuda
```

### 3.3 配置优先级

```
CLI 参数 > 环境变量 > YAML 配置 > 代码默认值
```

---

## 4. 训练数据格式

所有训练数据均为 **JSONL 格式**（每行一个 JSON 对象），默认放在 `data/finetune/` 目录下。

### 4.1 Embedding 微调 — 三元组格式

每条数据包含 query（查询）、positive（相关文档）、negative（不相关文档）。

```jsonl
{"query": "如何申请报销", "positive": "报销流程：填写申请表→部门审批→财务审核→打款", "negative": "公司年会将于12月举行"}
{"query": "年假怎么算", "positive": "员工入职满1年享有5天年假，10年以上享有10天", "negative": "会议室预约请使用OA系统"}
{"query": "加班费标准", "positive": "工作日加班按1.5倍工资计算，周末2倍，法定节假日3倍", "negative": "请保持办公区域整洁"}
```

**字段要求：** `query`、`positive`、`negative` 均为必需的非空字符串。

> **注意：** 训练时 `negative` 字段用于格式校验，但实际训练使用 `MultipleNegativesRankingLoss`（批次内负采样），不需要单独构造负例。`negative` 字段为后续扩展预留。

### 4.2 Reranker 微调 — 二分类格式

每条数据包含 query、document、label（1=相关，0=不相关）。

```jsonl
{"query": "python读取excel文件", "document": "使用pandas的read_excel方法可以读取Excel文件", "label": 1}
{"query": "python读取excel文件", "document": "Java使用POI库操作Excel文件", "label": 0}
{"query": "如何配置nginx", "document": "修改nginx.conf文件中的server块可以配置虚拟主机", "label": 1}
{"query": "如何配置nginx", "document": "Django的settings.py中配置数据库连接", "label": 0}
```

**字段要求：** `query` 和 `document` 为非空字符串；`label` 必须为 `0` 或 `1`（整数）。

### 4.3 LLM 微调（SFT / 蒸馏）— 指令格式

每条数据包含 instruction（指令）、input（输入）、output（期望输出）。

```jsonl
{"instruction": "根据以下文档内容回答问题", "input": "文档内容：报销流程如下...\n问题：报销需要哪些材料？", "output": "需要以下材料：1. 报销申请表 2. 发票原件 3. 审批单"}
{"instruction": "翻译以下内容为英文", "input": "公司年假制度：入职满1年享有5天年假", "output": "Annual leave policy: Employees with 1+ years of service receive 5 days of annual leave."}
{"instruction": "总结以下文档要点", "input": "（长文档内容...）", "output": "1. 核心流程... 2. 注意事项... 3. 相关联系人..."}
```

**字段要求：** `instruction` 和 `output` 为非空字符串；`input` 可为空字符串 `""`。

### 4.4 蒸馏数据（教师标签生成后）

蒸馏模式下，数据额外包含 `teacher_output` 字段：

```jsonl
{"instruction": "...", "input": "...", "output": "人工标注答案", "teacher_output": "云端大模型生成的答案"}
```

此字段由 `generate_teacher_labels()` 自动生成，无需手动编写。

---

## 5. CLI 操作

### 5.1 命令总览

```bash
python -m model.finetune {embedding|reranker|llm|list|info|remove} [选项]
```

### 5.2 Embedding 微调

```bash
# 基本用法
python -m model.finetune embedding --data data/finetune/triplets.jsonl

# 指定适配器名称和训练轮数
python -m model.finetune embedding \
    --data data/finetune/triplets.jsonl \
    --name my-embedding-v2 \
    --epochs 5 \
    --batch-size 16 \
    --lr 1e-4

# 使用自定义基座模型
python -m model.finetune embedding \
    --data data/finetune/triplets.jsonl \
    --base-model BAAI/bge-base-zh-v1.5 \
    --name bge-base-custom
```

### 5.3 Reranker 微调

```bash
python -m model.finetune reranker \
    --data data/finetune/rerank_data.jsonl \
    --name my-reranker-v1 \
    --epochs 3
```

### 5.4 LLM SFT 微调

```bash
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --name my-llm-sft-v1 \
    --epochs 5 \
    --batch-size 4
```

### 5.5 LLM 蒸馏（两步走）

**步骤 1：生成教师标签**

```bash
# 用云端大模型为数据集生成答案
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --teacher deepseek-v4-pro \

    --generate-only
```

> 输出文件: `data/finetune/instructions_with_teacher.jsonl`
>
> 此步骤支持**断点续传**：如果中途失败，重新运行相同命令即可从断点继续。

**步骤 2：训练学生模型**

```bash
python -m model.finetune llm \
    --data data/finetune/instructions_with_teacher.jsonl \
    --teacher deepseek-v4-pro \

    --alpha 0.3 \
    --name my-llm-distilled-v1
```

> `--alpha 0.3` 表示：30% 人工标注 + 70% 教师答案。`--alpha 0` = 纯蒸馏，`--alpha 1` = 纯 SFT。

**一键执行（生成 + 训练）：**

```bash
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --teacher deepseek-v4-pro \

    --alpha 0.3 \
    --name my-llm-distilled-v1
```

> 如果 `--data` 指向的文件已有 `teacher_output` 字段，则跳过生成步骤直接训练。

### 5.6 管理命令

```bash
# 列出所有已微调适配器
python -m model.finetune list
# 输出:
# 名称                            类型          基座模型                              创建时间
# my-embedding-v2                 embedding     BAAI/bge-large-zh-v1.5                 2026-07-12T14:30:00
# my-llm-distilled-v1             llm           Qwen/Qwen3-0.6B                        2026-07-12T15:00:00

# 查看适配器详情
python -m model.finetune info --name my-embedding-v2
# 输出:
# 名称:       my-embedding-v2
# 类型:       embedding
# 基座模型:   BAAI/bge-large-zh-v1.5
# 路径:       /project/local_models/finetuned/my-embedding-v2
# 创建时间:   2026-07-12T14:30:00
# 训练指标:   {'train_loss': 0.12, 'duration_seconds': 3600.5}
# 训练参数:   {'epochs': 5, 'learning_rate': 0.0001, 'batch_size': 16, ...}

# 删除适配器
python -m model.finetune remove --name my-embedding-v2
```

---

## 6. Python API 操作

### 6.1 基础微调

```python
from model import models

# === Embedding 微调 ===
result = models.finetune("embedding", data_path="data/finetune/triplets.jsonl")
print(result.adapter_path)   # → Path("local_models/finetuned/embedding_20260712_143000")
print(result.metrics)        # → {"train_loss": 0.12, "duration_seconds": 1200.5}

# 指定名称和参数
result = models.finetune(
    "embedding",
    data_path="data/finetune/triplets.jsonl",
    output_name="my-embedding-v3",
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
)

# === Reranker 微调 ===
result = models.finetune(
    "reranker",
    data_path="data/finetune/rerank_data.jsonl",
    output_name="my-reranker-v1",
)

# === LLM SFT 微调 ===
result = models.finetune(
    "llm",
    data_path="data/finetune/instructions.jsonl",
    output_name="my-llm-v1",
)
```

### 6.2 LLM 蒸馏

```python
from model import models

# 蒸馏训练
result = models.finetune(
    "llm",
    data_path="data/finetune/instructions.jsonl",
    teacher="deepseek-v4-pro",
    alpha=0.3,
    output_name="my-llm-distilled-v1",
)
```

### 6.3 自定义配置

```python
from model import models
from model.finetune.config import FinetuneConfig, TrainingConfig, LoRAConfig

# 构造自定义配置
config = FinetuneConfig(
    output_dir="local_models/finetuned",     # 或 Path 对象
    device="cuda",                      # 强制使用 GPU
    training=TrainingConfig(
        epochs=10,
        learning_rate=5e-5,
        batch_size=4,
        max_seq_length=1024,
    ),
    lora=LoRAConfig(
        r=16,
        lora_alpha=64,
    ),
)

result = models.finetune(
    "llm",
    data_path="data/finetune/instructions.jsonl",
    config=config,
    output_name="my-custom-llm",
)
```

### 6.4 管理已微调适配器

```python
from model import models

# 列出所有适配器
adapters = models.list_finetuned()
# → {"my-embedding-v3": FinetuneInfo(...), "my-llm-v1": FinetuneInfo(...)}

for name, info in adapters.items():
    print(f"{name}: {info.model_type} @ {info.adapter_path}")

# 获取适配器路径
path = models.get_finetuned_path("my-embedding-v3")
# → Path("local_models/finetuned/my-embedding-v3") 或 None

# 删除适配器
models.remove_finetuned("my-embedding-v3")
# → True（成功）/ False（不存在）
```

### 6.5 与下游模块集成

```python
from model import models
from sentence_transformers import SentenceTransformer

# 加载微调后的 Embedding 模型（基座 + LoRA 适配器）
adapter_path = models.get_finetuned_path("my-embedding-v3")
if adapter_path:
    # 方法 1：直接加载适配器目录（sentence-transformers >= 3.0）
    embedder = SentenceTransformer(str(adapter_path))
else:
    # 回退到基座模型
    base_path = models.get_path("embedding")
    embedder = SentenceTransformer(str(base_path))
```

---

## 7. 蒸馏工作流详解

### 7.1 原理

```
┌──────────────────────────────────────────────────┐
│  阶段 1: 教师标签生成（一次性，可离线）            │
│                                                   │
│  instructions.jsonl ──→ 云端大模型 API ──→ 教师答案 │
│  {instruction, input}    (DeepSeek/GPT)            │
│                                                   │
│  输出: instructions_with_teacher.jsonl             │
│    {instruction, input, output, teacher_output}    │
│                                                   │
│  阶段 2: 学生训练（本地 GPU）                      │
│                                                   │
│  hard_loss    = CrossEntropy(student, 人工标注)    │
│  distill_loss = CrossEntropy(student, 教师标注)    │
│  total_loss   = α × hard_loss + (1-α) × distill_loss │
│                                                   │
│  α = 0.0 → 纯蒸馏（只用教师答案）                  │
│  α = 1.0 → 纯 SFT（只用人工标注）                  │
│  α = 0.5 → 混合（各一半）                          │
└──────────────────────────────────────────────────┘
```

### 7.2 选择 alpha

| 场景 | 推荐 alpha | 说明 |
|------|-----------|------|
| 人工标注质量高、数量充足 | 0.7 ~ 1.0 | 以 SFT 为主 |
| 人工标注少、依赖大模型 | 0.3 ~ 0.5 | 均衡 |
| 无人工标注、纯蒸馏 | 0.0 | 完全依赖教师 |
| 教师质量存疑 | 0.8 ~ 1.0 | 人工标注权重更高 |

### 7.3 断点续传

`generate_teacher_labels()` 支持断点续传：

- 写入临时文件 `*.jsonl.tmp`，完成后原子替换
- 重新运行时从已有输出文件中恢复已完成的记录
- 按 `(instruction, input)` 复合键匹配，只处理未生成的记录

```bash
# 第一次运行（在中途失败）
python -m model.finetune llm --data data/instructions.jsonl --teacher claude-sonnet-5 --generate-only
# ... 处理了 300/1000 条后失败 ...

# 重新运行（自动从第 301 条继续）
python -m model.finetune llm --data data/instructions.jsonl --teacher claude-sonnet-5 --generate-only
# → 加载已有进度: 300 条 → 处理剩余 700 条
```

---

## 8. 配置调优参考

### 8.1 LoRA 参数

| 参数 | 默认值 | 调优建议 |
|------|--------|---------|
| `r` | 8 | 简单任务 4~8，复杂任务 16~32 |
| `lora_alpha` | 32 | 通常设为 `r` 的 2~4 倍 |
| `lora_dropout` | 0.1 | 数据少时可增至 0.2 防过拟合 |

### 8.2 训练参数

| 参数 | 默认值 | 调优建议 |
|------|--------|---------|
| `epochs` | 3 | Embedding/Reranker 2~5，LLM 3~10 |
| `learning_rate` | 2e-4 | LLM 微调可降至 5e-5 ~ 1e-4 |
| `batch_size` | 8 | 显存不足时减小 + 增大 `gradient_accumulation_steps` |
| `max_seq_length` | 512 | LLM 可增至 1024~2048（需更多显存） |

### 8.3 显存不足处理

```bash
# 减小批次 + 增大梯度累积
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --batch-size 2 \
    # 在 YAML 中设置 gradient_accumulation_steps: 8 等效 batch_size=16

# 使用 CPU（极慢，仅测试用）
python -m model.finetune embedding \
    --data data/finetune/triplets.jsonl \
    --device cpu
```

---

## 9. 故障排查

### 9.1 "无法确定基座模型"

```bash
# 显式指定 --base-model
python -m model.finetune reranker \
    --data data/rerank.jsonl \
    --base-model BAAI/bge-reranker-v2-m3
```

### 9.2 "数据格式错误"

检查 JSONL 文件是否满足对应格式要求（参见第 4 节）。常见问题：

- JSONL 每行必须是合法 JSON（不能有尾逗号）
- 字段名区分大小写 — `query` 不是 `Query`
- `label` 必须为整数 `0`/`1`，不能是字符串 `"0"`/`"1"`
- 空行会自动跳过，但空字段（如 `"output": ""`）会报错

### 9.3 蒸馏 API 调用失败

```bash
# 确认 .env 中设置了 API 密钥
echo $ANTHROPIC_API_KEY

# 网络问题 — 使用镜像端点
export HF_ENDPOINT=https://hf-mirror.com

# 教师 API 超时 — 修改 YAML 或在调用时捕获
```

### 9.4 CUDA Out of Memory

```bash
# 方案 1: 减小 batch_size
python -m model.finetune llm --data data/instructions.jsonl --batch-size 2

# 方案 2: 修改 YAML 增加梯度累积
# finetune.training.gradient_accumulation_steps: 8

# 方案 3: 使用更小的 LoRA rank
# finetune.lora.r: 4
```

### 9.5 `metadata.yaml` 损坏

```bash
# 手动删除损坏的适配器目录
rm -rf local_models/finetuned/corrupted-adapter-name/

# 然后 list 命令恢复正常
python -m model.finetune list
```

---

## 10. 最佳实践

1. **从小数据集开始** — 先用 50~100 条数据验证流程，确认 loss 收敛后再扩大
2. **监控 eval_loss** — 如果 eval_loss 持续上升而 train_loss 下降，说明过拟合，应减少 epochs 或增加 lora_dropout
3. **蒸馏先试 alpha=0.5** — 不确定时从 0.5 开始，观察教师答案质量再调整
4. **保留基座模型** — LoRA 适配器只有几 MB，不会覆盖基座模型，可以安全地保存多组适配器
5. **命名规范** — 建议使用 `{model_type}-{domain}-{version}` 格式，如 `embedding-finance-v2`
6. **记录训练参数** — 每个适配器的 `metadata.yaml` 自动记录了训练配置，可通过 `info` 命令查看
7. **定期清理** — 用 `list` 检查，用 `remove` 删除不再需要的适配器
