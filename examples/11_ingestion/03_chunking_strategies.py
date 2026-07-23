"""
03_chunking_strategies.py — 文档分块：三种策略对比

演示内容：
  1. 加载演示文本（优先复用 01 解析产物，独立运行时内嵌降级）
  2. 三种分块策略执行 — FixedWindow / Hierarchical / Semantic
  3. 结果对比表（分块数、平均大小、heading 覆盖率）
  4. 每种策略展示前 2 个 chunk

运行方式：
  cd rag0709
  python examples/11_ingestion/03_chunking_strategies.py

前置条件: Semantic 策略需 Embedding 模型（缺失时自动跳过）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples._common import banner, check_embedding_model  # noqa: E402

from config import settings  # noqa: E402


# ── 演示用文本（当 01 解析产物不可用时使用）────────────────────────
FALLBACK_TEXT = """# 员工手册

## 第一章 总则

本手册适用于公司全体员工，旨在规范公司管理，保障员工权益。公司秉持"以人为本、诚信经营"的理念，为员工提供良好的工作环境和发展平台。

## 第二章 薪资福利

### 2.1 薪资结构

员工薪资由基本工资、岗位工资和绩效工资三部分组成。每月15日发放上月工资，遇节假日顺延。

### 2.2 五险一金

公司为所有正式员工缴纳社会保险（养老、医疗、失业、工伤、生育）和住房公积金。缴费基数按员工上年度月平均工资确定。

## 第三章 休假制度

### 3.1 带薪年休假

工作满1年不满10年的员工，每年享受5天带薪年假。工作满10年不满20年的员工，每年享受10天带薪年假。工作满20年以上的员工，每年享受15天带薪年假。

年假可累积至次年3月31日，逾期未休视为自动放弃。

### 3.2 病假

员工因病需要休息的，应持医院出具的病假证明。病假期间工资按国家规定执行。

### 3.3 事假

事假需提前申请，经部门主管审批。事假期间不计发工资。

## 第四章 费用报销

### 4.1 差旅费

出差需提前填写出差申请单，经部门主管审批。差旅费用在出差结束后一周内报销。住宿标准按城市等级划分。

### 4.2 交通费

因公出行产生的交通费用，凭票据实报实销。市内交通优先使用公共交通工具。

### 4.3 加班餐补

工作日加班超过2小时，可报销餐费补贴30元/次。周末加班超过4小时，可报销餐费补贴50元/次。
"""


def _load_text() -> str:
    """加载文本：优先复用 01 解析产物，否则使用内嵌文本"""
    parsed_dir = settings.ingestion.parsed_doc_dir
    if parsed_dir.exists():
        md_files = sorted(parsed_dir.glob("*.md"))
        if md_files:
            path = md_files[0]
            print(f"  📄 复用解析产物: {path.name} ({path.stat().st_size:,} bytes)")
            return path.read_text(encoding="utf-8")

    print(f"  📄 使用内嵌文本 ({len(FALLBACK_TEXT):,} 字符)")
    print(f"  提示: 先运行 01_document_parsing.py 解析真实文档")
    return FALLBACK_TEXT


def _print_chunk_preview(chunks, strategy_name: str, max_show: int = 2) -> None:
    """打印分块预览"""
    print(f"\n  ── {strategy_name} ──")
    for i, c in enumerate(chunks[:max_show], 1):
        preview = c.text[:120].replace("\n", " ")
        heading = c.metadata.get("heading_path", "")
        extra = f" [{heading}]" if heading else ""
        print(f"    [{i}] {len(c.text):4d} chars{extra}")
        print(f"        {preview}...")
    if len(chunks) > max_show:
        print(f"    ... 共 {len(chunks)} 个 chunk")


async def main():
    # ── 1. 加载文本 ───────────────────────────────────────────────
    banner("1. 加载演示文本")
    text = _load_text()

    cfg = settings.chunking
    print(f"\n  分块配置: chunk_size={cfg.chunk_size}, overlap={cfg.overlap}")

    # ── 2. FixedChunker ───────────────────────────────────────────
    banner("2. FixedChunker — 固定窗口分块")

    from ingestion.chunker import FixedChunker

    fixed = FixedChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    fixed_chunks = fixed.split(text)
    print(f"  分块数: {len(fixed_chunks)}")
    if fixed_chunks:
        sizes = [len(c.text) for c in fixed_chunks]
        print(f"  平均大小: {sum(sizes)/len(sizes):.0f} chars")
        print(f"  最小/最大: {min(sizes)} / {max(sizes)} chars")
    _print_chunk_preview(fixed_chunks, "FixedChunker")

    # ── 3. HierarchicalChunker ────────────────────────────────────
    banner("3. HierarchicalChunker — 层级分块")

    from ingestion.chunker import HierarchicalChunker

    hier = HierarchicalChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    hier_chunks = hier.split(text)
    print(f"  分块数: {len(hier_chunks)}")
    if hier_chunks:
        sizes = [len(c.text) for c in hier_chunks]
        print(f"  平均大小: {sum(sizes)/len(sizes):.0f} chars")
        print(f"  最小/最大: {min(sizes)} / {max(sizes)} chars")
        # heading_path 覆盖率
        with_heading = sum(1 for c in hier_chunks if c.metadata.get("heading_path"))
        print(f"  heading_path 覆盖: {with_heading}/{len(hier_chunks)} ({100*with_heading//len(hier_chunks)}%)")
    _print_chunk_preview(hier_chunks, "HierarchicalChunker")

    # ── 4. SemanticChunker（需 embedding 模型）─────────────────────
    banner("4. SemanticChunker — 语义边界分块")

    has_model = check_embedding_model(auto_download=False)
    if not has_model:
        print("\n  ⚠️  SemanticChunker 需要 Embedding 模型，跳过。")
        print("  下载命令: models.download('embedding')")
        sem_chunks = []
    else:
        from model import models
        from ingestion.chunker import SemanticChunker

        sem = SemanticChunker(
            embedding_model=models.embedding_model,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
            threshold_percentile=cfg.semantic_threshold_percentile,
            buffer_size=cfg.semantic_buffer_size,
        )
        sem_chunks = sem.split(text)
        print(f"  分块数: {len(sem_chunks)}")
        if sem_chunks:
            sizes = [len(c.text) for c in sem_chunks]
            print(f"  平均大小: {sum(sizes)/len(sizes):.0f} chars")
            print(f"  最小/最大: {min(sizes)} / {max(sizes)} chars")
            for c in sem_chunks:
                print(c.text.replace("\n", " "))
        _print_chunk_preview(sem_chunks, "SemanticChunker")

    # ── 5. 三种策略对比汇总 ───────────────────────────────────────
    banner("5. 三种策略对比汇总")

    # 表头
    print(f"  {'策略':<20s} {'分块数':>6s} {'平均大小':>8s} {'最小':>6s} {'最大':>6s}")
    print(f"  {'-'*52}")

    for name, chunks in [
        ("FixedChunker", fixed_chunks),
        ("HierarchicalChunker", hier_chunks),
        ("SemanticChunker", sem_chunks),
    ]:
        if chunks:
            sizes = [len(c.text) for c in chunks]
            print(f"  {name:<20s} {len(chunks):>6d} {sum(sizes)/len(sizes):>7.0f} {min(sizes):>6d} {max(sizes):>6d}")
        else:
            print(f"  {name:<20s} {'(跳过)':>6s}")

    print(f"\n  配置: chunk_size={cfg.chunk_size}, overlap={cfg.overlap}")
    print(f"  推荐: 层级清晰的文档用 HierarchicalChunker，通用文档用 SemanticChunker")

    banner("✅ 分块策略对比演示完成")
    print()
    print("  下一步: 04_full_pipeline.py — 完整 Pipeline + 索引验证")


if __name__ == "__main__":
    asyncio.run(main())
