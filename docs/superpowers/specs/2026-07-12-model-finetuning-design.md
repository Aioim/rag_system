# 模型微调 & 蒸馏 — 设计文档

> 日期：2026-07-12
> 状态：设计完成，待评审
> 关联文档：[[2026-07-09-rag-enterprise-qa-design.md]]

---

## 1. 需求概述

在现有 `src/model/` 模型下载管理能力之上，增加模型微调与知识蒸馏能力。覆盖 Embedding、Reranker、LLM 三种模型类型的微调，LLM 额外支持黑盒蒸馏（云端大模型作为教师）。

| 维度 | 决策 |
|------|------|
| 范围 | Embedding + Reranker + LLM，全部支持微调 |
| 微调方式 | LoRA 为主（轻量适配器），可选全量微调 |
| 蒸馏 | 仅 LLM，教师模型为**云端 API**（Claude/GPT 等），走黑盒蒸馏 |
| 触发方式 | CLI（`python -m model.finetune`）+ Python API（`models.finetune()`） |
| 训练数据 | 第一阶段只支持外部 JSONL 导入，从反馈日志自动构建留给后续迭代 |
| 教师模型 | 仅云端 API（Claude Sonnet 5 等），走项目现有 LLM 路由 |

---

## 2. 模块结构

```
src/model/
├── __init__.py                # 新增导出 finetune 相关
├── downloader.py              # 已有（不变）
├── manager.py                 # 已有 + 新增 finetune/list_finetuned/get_finetuned_path/remove_finetuned
├── README.md                  # 更新
└── finetune/
    ├── __init__.py             # 导出所有 Trainer + CLI 入口
    ├── config.py               # FinetuneConfig — Pydantic 配置模型
    ├── base.py                 # BaseTrainer — 抽象基类（日志/进度/回调/保存）
    ├── data.py                 # 数据加载 & 验证（JSONL → Dataset）
    ├── embedding_trainer.py    # EmbeddingTrainer — 基于 sentence-transformers 微调
    ├── reranker_trainer.py     # RerankerTrainer — CrossEncoder 微调
    ├── llm_trainer.py          # LLMTrainer — SFT 微调 + 蒸馏模式
    └── cli.py                  # argparse CLI 入口（python -m model.finetune）
```

### 适配器文件组织

```
models/finetuned/
├── my-embedding-v1/              # embedding LoRA 适配器
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── metadata.yaml             # {model_type, base_model, created_at, metrics, training_config}
├── my-reranker-v1/               # reranker LoRA 适配器
│   └── ...
└── my-llm-distilled/             # LLM LoRA 适配器
    └── ...
```

---

## 3. 配置设计

### defaults.yaml 新增段

```yaml
finetune:
  output_dir: models/finetuned       # LoRA 适配器输出目录（相对 PROJECT_ROOT）
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
    r: 8
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: null            # null=各模型类型自动推断

  distillation:
    temperature: 2.0                # 软化教师分布（当前版本仅保留配置，黑盒蒸馏下影响较小）
    alpha: 0.5                      # 硬标签 vs 教师标签损失权重（0=纯蒸馏, 1=纯SFT）
```

### Pydantic 模型（`config.py`）

```python
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Literal

class LoRAConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[list[str]] = None

class TrainingConfig(BaseModel):
    epochs: int = 3
    learning_rate: float = 2.0e-4
    batch_size: int = 8
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50

class DistillationConfig(BaseModel):
    temperature: float = 2.0
    alpha: float = 0.5

class FinetuneConfig(BaseModel):
    output_dir: Path
    device: Literal["auto", "cuda", "cpu"] = "auto"
    data_dir: Path = Path("data/finetune")
    training: TrainingConfig = TrainingConfig()
    lora: LoRAConfig = LoRAConfig()
    distillation: DistillationConfig = DistillationConfig()
```

---

## 4. 训练数据格式（JSONL）

### Embedding — 三元组格式

```jsonl
{"query": "如何申请报销", "positive": "报销流程：填写申请表→部门审批→财务审核→打款", "negative": "公司年会将于12月举行"}
{"query": "年假怎么算", "positive": "员工入职满1年享有5天年假...", "negative": "会议室预约请使用OA系统"}
```

### Reranker — 二分类格式

```jsonl
{"query": "python读取excel", "document": "使用pandas的read_excel方法可以读取Excel文件", "label": 1}
{"query": "python读取excel", "document": "Java使用POI库操作Excel文件", "label": 0}
```

### LLM — 指令格式

```jsonl
{"instruction": "根据以下文档内容回答问题", "input": "文档内容：报销流程如下...\n问题：报销需要哪些材料？", "output": "需要以下材料：1. 报销申请表 2. 发票原件 3. 审批单"}
```

### 蒸馏用数据（教师生成后）

```jsonl
{"instruction": "...", "input": "...", "output": "人工标注答案", "teacher_output": "教师模型生成的答案"}
```

`data.py` 提供统一的加载、schema 验证、train/eval 切分（默认 8:2）。

---

## 5. 核心组件设计

### 5.1 BaseTrainer（`base.py`）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datasets import Dataset

@dataclass
class FinetuneResult:
    model_type: str              # "embedding" | "reranker" | "llm"
    base_model: str              # 基座模型 repo_id
    adapter_path: Path           # LoRA 适配器路径
    output_name: str             # 用户指定的名称
    metrics: dict                # {"train_loss": 0.12, "eval_loss": 0.15, "epochs": 3}
    duration_seconds: float

class BaseTrainer(ABC):
    """模型微调基类"""

    def __init__(self, config: FinetuneConfig, base_model_id: str):
        self.config = config
        self.base_model_id = base_model_id  # HuggingFace repo_id
        self._output_name: Optional[str] = None

    @abstractmethod
    def load_data(self, data_path: Path) -> tuple[Dataset, Optional[Dataset]]: ...

    @abstractmethod
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset]) -> Path: ...

    def run(self, data_path: Path, output_name: Optional[str] = None) -> FinetuneResult:
        """完整流程：验证数据 → 加载 → 训练 → 保存 → 返回结果"""
        self._output_name = output_name or self._generate_default_name()
        self._validate_data(data_path)
        train_ds, eval_ds = self.load_data(data_path)
        adapter_path = self.train(train_ds, eval_ds)
        return FinetuneResult(
            model_type=self.model_type,
            base_model=self.base_model_id,
            adapter_path=adapter_path,
            output_name=self._output_name,
            metrics=self._collect_metrics(),
            duration_seconds=self._duration,
        )

    def _resolve_device(self) -> str:
        """解析 device 配置：auto → 检测 CUDA → 回退 CPU"""
        ...

    def _get_output_dir(self) -> Path:
        """返回 LoRA 适配器输出目录"""
        return self.config.output_dir / self._output_name

    def _validate_data(self, data_path: Path) -> None: ...
    def _save_metadata(self, result: FinetuneResult) -> None: ...
```

### 5.2 EmbeddingTrainer（`embedding_trainer.py`）

```python
class EmbeddingTrainer(BaseTrainer):
    """微调 BGE Embedding 模型

    基座模型: BAAI/bge-large-zh-v1.5
    损失函数: MultipleNegativesRankingLoss
    框架: sentence-transformers
    """

    model_type = "embedding"

    def load_data(self, data_path: Path):
        # JSONL → InputExample(texts=[query, positive, negative])
        # 8:2 train/eval split
        examples = []
        for line in read_jsonl(data_path):
            examples.append(InputExample(
                texts=[line["query"], line["positive"], line["negative"]]
            ))
        return split_train_eval(examples)

    def train(self, train_dataset, eval_dataset):
        # 1. 加载基座模型 → SentenceTransformer(base_model_id)
        # 2. 注入 LoRA → get_peft_model()
        # 3. 构建 MultipleNegativesRankingLoss
        # 4. SentenceTransformerTrainer → 训练
        # 5. 保存 LoRA 适配器到 output_dir
```

### 5.3 RerankerTrainer（`reranker_trainer.py`）

```python
class RerankerTrainer(BaseTrainer):
    """微调 BGE-Reranker CrossEncoder 模型

    基座: BAAI/bge-reranker-v2-m3
    损失: CrossEntropyLoss（相关/不相关二分类）
    """

    model_type = "reranker"

    def load_data(self, data_path: Path):
        # JSONL → CrossEncoder InputExample(texts=[query, document], label=0/1)
        # 8:2 train/eval split

    def train(self, train_dataset, eval_dataset):
        # 1. 加载基座 CrossEncoder
        # 2. 注入 LoRA → get_peft_model()
        # 3. HuggingFace Trainer + CrossEntropyLoss
        # 4. 保存 LoRA 适配器
```

### 5.4 LLMTrainer — SFT + 黑盒蒸馏（`llm_trainer.py`）

```python
class LLMTrainer(BaseTrainer):
    """LLM 微调/蒸馏

    基座: Qwen3-0.6B (默认)
    教师: 云端 API（Claude/GPT 等），通过项目 LLM 路由调用
    """

    model_type = "llm"

    def __init__(self, config: FinetuneConfig, base_model_id: str,
                 teacher_model: Optional[str] = None):
        super().__init__(config, base_model_id)
        self.teacher_model = teacher_model  # None=纯SFT, 非None=蒸馏模式

    def generate_teacher_labels(self, data_path: Path) -> Path:
        """第1步(蒸馏): 用教师模型为每条数据生成答案

        输入: instructions.jsonl (含 instruction + input)
        输出: instructions_with_teacher.jsonl (额外增加 teacher_output 字段)
        策略: 批处理 + API 调用失败自动重试 + 断点续传
        """
        ...

    def train(self, train_dataset, eval_dataset):
        """第2步: 训练学生模型

        SFT模式（teacher_model=None）:
          loss = CrossEntropy(student_logits, labels)

        蒸馏模式（teacher_model 不为 None）:
          hard_loss = CrossEntropy(student_logits, 人工标注)
          distill_loss = CrossEntropy(student_logits, 教师标注)
          total_loss = alpha * hard_loss + (1-alpha) * distill_loss

        其中 alpha 来自 config.distillation.alpha
        """
```

### 5.5 蒸馏流程

```
┌─────────────────────────────────────────────────┐
│  1. 数据准备阶段（训练前，一次性完成）              │
│                                                  │
│   原始指令数据 ──→ 云端教师模型 ──→ 教师答案        │
│     {instruction, input}     (Claude API)         │
│                                                  │
│   最终训练数据:                                    │
│     {instruction, input, output, teacher_output}  │
│                                                  │
│  2. 训练阶段（本地）                               │
│                                                  │
│    hard_loss = CrossEntropy(student, human_label) │
│    distill_loss = CrossEntropy(student, teacher_label) │
│    total_loss = α × hard_loss + (1-α) × distill_loss │
│                                                  │
│    纯蒸馏模式（α=0）：只用教师答案训练              │
│    SFT模式（α=1）：只用人工标注训练                 │
│    混合模式（0<α<1）：两者都用                      │
└─────────────────────────────────────────────────┘
```

教师模型通过项目 LLM 路由模块调用（预留接口，依赖 `src/generation/` 尚未实现，初期可通过环境变量直接配置 API key 和 endpoint 作为过渡方案）。

---

## 6. CLI & Python API

### 6.1 CLI

```bash
# --- Embedding 微调 ---
python -m model.finetune embedding \
    --data data/finetune/triplets.jsonl \
    --name my-embedding-v1 \
    --epochs 5

# --- Reranker 微调 ---
python -m model.finetune reranker \
    --data data/finetune/rerank_data.jsonl \
    --name my-reranker-v1

# --- LLM SFT 微调 ---
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --name my-llm-v1

# --- LLM 蒸馏（教师生成 + 训练） ---
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --name my-llm-distilled \
    --teacher claude-sonnet-5 \
    --alpha 0.3

# --- LLM 蒸馏（分步：只生成教师答案，不训练） ---
python -m model.finetune llm \
    --data data/finetune/instructions.jsonl \
    --teacher claude-sonnet-5 \
    --generate-only

# --- 管理命令 ---
python -m model.finetune list                          # 列出所有已微调适配器
python -m model.finetune remove --name my-llm-v1       # 删除指定适配器
python -m model.finetune info --name my-llm-v1         # 查看适配器详情（基座模型/训练参数/指标）
```

CLI 参数优先级：**命令行参数 > 环境变量 > YAML 配置 > 代码默认值**（与现有配置体系一致）。

### 6.2 Python API

```python
from model import models
from model.finetune import FinetuneConfig, TrainingConfig

# --- 基础微调 ---
result = models.finetune("embedding", data_path="data/finetune/triplets.jsonl")
# result.metrics      → {"train_loss": 0.12, "eval_loss": 0.15, "epochs": 3}
# result.adapter_path  → Path("models/finetuned/embedding_20260712_143000")

result = models.finetune("reranker", data_path="data/finetune/rerank_data.jsonl",
                         output_name="my-reranker")

# --- LLM 蒸馏 ---
result = models.finetune("llm", data_path="data/finetune/instructions.jsonl",
                         teacher="claude-sonnet-5", alpha=0.3)

# --- 自定义配置 ---
config = FinetuneConfig(
    training=TrainingConfig(epochs=5, learning_rate=1e-4, batch_size=4)
)
result = models.finetune("embedding", data_path="...", config=config)

# --- 管理接口 ---
models.list_finetuned()              # → {"my-llm-v1": FinetuneInfo(...), ...}
models.get_finetuned_path("my-llm")  # → Path or None
models.remove_finetuned("my-llm")    # → bool
```

### 6.3 FinetuneInfo

```python
@dataclass
class FinetuneInfo:
    name: str                # 适配器名称
    model_type: str          # "embedding" | "reranker" | "llm"
    base_model: str          # 基座模型 repo_id
    adapter_path: Path       # 本地路径
    created_at: str          # ISO 时间戳
    metrics: dict            # 训练指标
    training_config: dict    # 训练参数快照
```

---

## 7. ModelManager 扩展

在现有 `ModelManager` 单例上新增以下方法：

```python
class ModelManager:
    # ... 现有方法不变 ...

    def finetune(self, model_type: str, data_path: str,
                 output_name: Optional[str] = None,
                 teacher: Optional[str] = None,
                 config: Optional[FinetuneConfig] = None,
                 **overrides) -> FinetuneResult:
        """微调指定类型的模型"""
        ...

    def list_finetuned(self) -> dict[str, FinetuneInfo]:
        """列出所有已微调的适配器"""
        ...

    def get_finetuned_path(self, name: str) -> Optional[Path]:
        """获取微调适配器路径"""
        ...

    def remove_finetuned(self, name: str) -> bool:
        """删除微调适配器"""
        ...
```

### 与下游模块的集成（后续）

```python
# 使用微调后的 Embedding 模型
path = models.get_finetuned_path("my-embedding-v1")
if path:
    embedder = SentenceTransformer(str(path))
else:
    embedder = SentenceTransformer(str(models.get_path("embedding")))
```

---

## 8. 依赖

```bash
# 新增
pip install peft>=0.12.0          # LoRA 微调
pip install datasets>=2.19.0      # 数据集加载
pip install accelerate>=0.30.0    # 训练加速
pip install sentence-transformers>=3.0.0  # Embedding 训练（已隐式依赖）

# 已有
# huggingface_hub (download)
# torch (已通过 sentence-transformers 安装)
```

---

## 9. 与现有模块的关系

```
现有能力:
  models.download("embedding")  → 下载基座模型
  models.get_path("embedding")   → 获取基座模型路径

新增能力:
  models.finetune("embedding", ...)  → 微调基座模型 → 输出 LoRA 适配器
  models.get_finetuned_path("my-lora") → 获取适配器路径
  models.list_finetuned()            → 列出所有适配器

完整模型生命周期:
  下载基座模型 → 微调（生成 LoRA 适配器）→ 下游加载（基座 + 适配器）→ 推理
```

---

## 10. 未尽事项 & 后续迭代

以下不在本期范围，设计已预留扩展点：

- **反馈驱动的自动数据集构建**：从用户 👍/👎 反馈 + 对话日志中半自动生成训练数据（策略全景图 §22）
- **全量微调**（Full Fine-tuning）：当前只做 LoRA，全量微调作为可选模式预留
- **Unsloth 加速后端**：作为可选训练引擎替换 HuggingFace Trainer
- **训练任务队列化**：CLI 当前是同步执行，后续可接入 Celery/ARQ 做异步训练 + 进度查询
- **A/B 评估框架**：微调前后检索/生成质量的自动化对比评测
- **教师标签缓存**：同一份数据集 + 同一教师模型的生成结果可缓存复用，避免重复调用 API
