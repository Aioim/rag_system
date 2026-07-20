# Ingestion 模块 — 离线文档处理 Pipeline

## 模块概述

Ingestion 模块负责将原始文档（PDF/Word/Markdown 等）解析、分块、嵌入后持久化到 FAISS 索引，供在线 retrieval 模块检索。

- **可插拔解析器**：Docling / PyMuPDF4LLM / MinerU / Direct，按文件扩展名自动选择
- **三种分块策略**：语义 / 固定窗口 / Markdown 层级
- **幂等写入**：同 doc_id 重复写入时自动替换旧向量
- **模型共享**：Embedding 模型跨调用缓存，避免重复加载

## 文件结构

```
ingestion/
├── __init__.py          # 导出 + create_default_pipeline 工厂
├── __main__.py          # CLI 演示入口 (python -m ingestion)
├── pipeline.py          # IngestionPipeline — Stage 编排器
├── stage.py             # Stage 协议（name/fatal/run）
├── context.py           # PipelineContext / StageError 数据容器
├── parser.py            # ParserStage — 委托至可插拔解析器
├── chunker.py           # ChunkerStage — 三种分块策略
├── embedder.py          # EmbedderStage — 批量 embedding
├── indexer.py           # FAISSIndexWriter — 索引持久化
└── parsers/             # 可插拔解析器
    ├── __init__.py      # 注册表 + get_parser 工厂
    ├── base.py          # BaseParser 抽象基类
    ├── docling_parser.py       # Docling — PDF/Word/PPT/HTML
    ├── pymupdf4llm_parser.py   # PyMuPDF4LLM — PDF
    ├── mineru_parser.py        # MinerU — PDF（复杂排版优化）
    └── direct_parser.py        # 直接读取 .md/.txt
```

## 快速开始

```python
from ingestion import create_default_pipeline

pipeline = create_default_pipeline()

# 处理文档并写入索引
ctx = await pipeline.run("docs/员工手册.pdf", collection="hr_docs")
# → ctx.document        原始文档信息
# → ctx.chunks          生成的分块（含 embedding）
# → ctx.status          "done" / "failed"
# → ctx.errors          非致命错误列表
# → ctx.metadata        各阶段耗时 (parser_ms / chunker_ms / embedder_ms)
```

### CLI 演示

```bash
python -m ingestion  # 处理内置示例文档
```

## Pipeline 流程

```
Parser → Chunker → Embedder → FAISSIndexWriter
  │         │          │             │
  │    SemanticChunker  │     写入向量 + docstore
  │    FixedWindow       │     (同 doc_id 自动替换)
  │    Hierarchical      │
  │                   批量编码
  │                   (幂等：已嵌入的跳过)
  │
解析器路由（按 file_type）
  ├── pdf → docling / pymupdf4llm / mineru
  ├── docx → docling
  ├── md → direct
  └── ...
```

## 解析器配置

```yaml
# config/{env}.yaml
ingestion:
  parsed_doc_dir: data/parsed_docs       # 解析后 .md 文件目录
  parsers:
    pdf: docling                          # pdf: docling | pymupdf4llm | mineru
    md: direct
    txt: direct
  mineru:                                 # MinerU 专用配置
    device: cpu
    models_dir: local_models/mineru
```

## Chunker 策略

| 策略 | 实现 | 适用场景 |
|------|------|----------|
| SemanticChunker | 语义相似度切分 | 通用文档，自适应语义边界 |
| FixedWindowChunker | 固定窗口 + 滑动步长 | 结构化较弱的文本 |
| HierarchicalChunker | 按 Markdown 标题层级 | 层级清晰的文档 |

## 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `chunking.chunk_size` | 512 | 分块大小 |
| `chunking.overlap` | 64 | 分块重叠 |
| `chunking.strategy` | semantic | 分块策略 |
| `chunking.semantic_threshold_percentile` | 0.9 | 语义切分敏感度 |
| `embedding.model` | BAAI/bge-large-zh-v1.5 | Embedding 模型 |
| `embedding.device` | cpu | 设备 |
| `embedding.batch_size` | 32 | 编码批量大小 |
| `faiss.index_type` | IVF_FLAT | FAISS 索引类型 |
| `faiss.metric_type` | COSINE | 相似度度量 |

## 依赖

```bash
pip install sentence-transformers faiss-cpu docling
# MinerU 解析器（可选）
pip install magic-pdf[full-cpu] ultralytics doclayout-yolo rapidocr-onnxruntime rapid-table
```
