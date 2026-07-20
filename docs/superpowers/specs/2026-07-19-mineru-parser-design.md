# MinerU 解析器 — 设计文档

**日期**: 2026-07-19
**状态**: 设计中

## 目标

在 `src/ingestion/parsers/` 模块中新增 MinerU 解析器后端，使用户可通过配置将 PDF 解析从 docling 切换至 MinerU。

## 动机

MinerU 在以下场景优于 docling：
- 复杂排版的 PDF（学术论文、技术手册）
- 图文混排文档中的公式/表格识别
- 中文文档的版面分析

## 架构

```
src/ingestion/parsers/
├── base.py                  # 不变
├── direct_parser.py         # 不变
├── docling_parser.py        # 不变
├── pymupdf4llm_parser.py    # 不变
├── mineru_parser.py         # ✨ 新增
└── __init__.py              # 注册 mineru → 注册表 + 导出
```

### MinerUParser

- 继承 `BaseParser`
- `name = "mineru"`
- `supported_formats = ("pdf",)`
- 使用 MinerU v1.1+ `PymuDocDataset` API
- 延迟导入 + 双检锁线程安全（与 DoclingParser 一致）
- `parse(source_path, output_dir=None)` 返回 Markdown 文本（图片已写入磁盘，Markdown 中相对路径引用）

### 接口变更

`BaseParser.parse()` 签名增加可选参数（向后兼容）：

```python
def parse(self, source_path: Path, output_dir: Path | None = None) -> str:
```

- `output_dir` — 图片/Markdown 产物的输出目录。`DirectParser`/`DoclingParser`/`PyMuPDF4LLMParser` 忽略此参数
- `ParserStage` 传入 `parsed_doc_dir` 作为 `output_dir`

### 调用流程

```
parse(source_path: Path, output_dir: Path) → str
  1. 延迟导入 magic_pdf 相关模块（双检锁）
  2. 读取 PDF bytes
  3. 创建 PymuDocDataset
  4. 设置图片输出目录: output_dir/{stem}_images/
  5. 执行分类 → 解析 → 生成 Markdown（图片写入上述目录）
  6. 返回 Markdown 文本（图片以相对路径引用: {stem}_images/img_001.png）
  7. ParserStage 将返回的 Markdown 写入 output_dir/{doc_id}.md
```

### 图片处理

- MinerU 解析过程中提取的图片写入 `parsed_doc_dir/{doc_id}_images/`
- Markdown 中通过相对路径 `{doc_id}_images/img_001.png` 引用
- 与 .md 文件同级，天然可解析
- `doc_id` 由 `source_path.stem` 推导

## 配置

### `config/defaults.yaml` 新增

```yaml
ingestion:
  mineru:
    device: cpu                  # cpu | cuda | mps
    models_dir: local_models/mineru    # MinerU 模型权重目录
```

### 切换方式

用户修改 `ingestion.parsers.pdf` 即可：

```yaml
ingestion:
  parsers:
    pdf: mineru                  # 从 docling 切换为 mineru
```

## 数据模型

`src/config/settings.py` 新增 Pydantic 模型：

```python
class MinerUConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    models_dir: str = "local_models/mineru"

class IngestionConfig(BaseModel):
    # ... 现有字段
    mineru: MinerUConfig = Field(default_factory=MinerUConfig)
```

## 依赖

- `magic-pdf[full-cpu]` 或 `magic-pdf[full]`（可选依赖，运行时按需安装）
- 不新增 `requirements.txt` 强制依赖，`import magic_pdf` 失败时给出安装提示

## 测试

| 测试 | 类型 | 说明 |
|------|------|------|
| 注册表可见 | 单元 | `get_parser("mineru")` 返回 MinerUParser 实例 |
| 懒加载 | 单元 | 在无 magic_pdf 环境下实例化不报错，parse() 时才报 ImportError |
| Mock 解析 | 单元 | mock `PymuDocDataset`，验证 Markdown 输出和图片路径 |
| 配置读取 | 单元 | 验证 `settings.ingestion.mineru.device` 等默认值 |
| 端到端 | 集成 | 用真实 PDF 通过 MinerU 解析 → 分块 → 入索引（可选，需 mineru 环境） |

## 风险

| 风险 | 缓解 |
|------|------|
| MinerU API 版本兼容 | 延迟导入 + clear ImportError 提示，不影响不使用 MinerU 的用户 |
| 模型下载大 | 模型目录独立于 embedding/reranker，用户可选安装 |
| 解析耗时 | ParserStage 已通过 `run_in_executor` 异步化，不阻塞事件循环 |

## 变更文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/ingestion/parsers/base.py` | 修改 | `parse()` 签名增加可选 `output_dir` 参数 |
| `src/ingestion/parsers/mineru_parser.py` | 新增 | MinerUParser 实现 |
| `src/ingestion/parsers/__init__.py` | 修改 | 注册 mineru，新增导出 |
| `src/ingestion/parser.py` | 修改 | `ParserStage` 传入 `output_dir` 给解析器 |
| `src/config/settings.py` | 修改 | 新增 MinerUConfig，挂载到 IngestionConfig |
| `config/defaults.yaml` | 修改 | 新增 `ingestion.mineru` 配置段 |
| `tests/unit/ingestion/test_mineru_parser.py` | 新增 | MinerU 解析器单元测试 |
| `CLAUDE.md` | 修改 | 更新解析器后端列表 |
