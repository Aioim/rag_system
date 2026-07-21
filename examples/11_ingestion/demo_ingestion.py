"""
demo_ingestion.py — 文档处理模块演示

演示内容：
  1. IngestionPipeline 组装
  2. 解析器选择（docling / pymupdf4llm / mineru / direct）
  3. 三种分块策略（semantic / fixed / hierarchical）
  4. 文档处理完整流程
  5. FAISS 索引持久化

运行方式：
  cd rag0709
  python examples/11_ingestion/demo_ingestion.py

前置条件：
  Embedding 模型需已下载: models.download('embedding')
  首次运行会触发 lazy download，耗时取决于网络。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings, PROJECT_ROOT  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    # ── 1. 检查 Embedding 模型 ──────────────────────────────────
    banner("1. 检查依赖")

    from model import models

    emb_status = models.status()
    print(f"  Embedding 模型: {'✅ 已下载' if emb_status.get('embedding') else '⬜ 未下载'}")
    if not emb_status.get("embedding"):
        print("  正在下载 Embedding 模型（首次运行，耗时较长）...")
        try:
            models.download("embedding")
            print("  ✅ 下载完成")
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            print("  请手动下载: models.download('embedding')")
            print("  或设置 HUGGINGFACE_TOKEN 环境变量后重试")
            return

    # ── 2. 创建演示文档 ─────────────────────────────────────────
    banner("2. 创建演示文档")

    tmp_dir = Path(tempfile.mkdtemp())
    demo_md = tmp_dir / "员工手册_示例.md"

    demo_content = """# 员工手册

## 第一章 总则

本手册适用于公司全体员工，旨在规范公司管理，保障员工权益。

## 第二章 薪资福利

### 2.1 薪资结构

员工薪资由基本工资、岗位工资和绩效工资三部分组成。每月15日发放上月工资。

### 2.2 五险一金

公司为所有正式员工缴纳社会保险（养老、医疗、失业、工伤、生育）和住房公积金。

### 2.3 年终奖金

根据公司年度经营业绩和个人绩效考核结果，于每年1月发放上年度年终奖金。

## 第三章 休假制度

### 3.1 带薪年休假

工作满1年不满10年的员工，每年享受5天带薪年假。
工作满10年不满20年的员工，每年享受10天带薪年假。
工作满20年以上的员工，每年享受15天带薪年假。

年假可累积至次年3月31日，逾期未休视为自动放弃。

### 3.2 病假

员工因病需要休息的，应持医院出具的病假证明。病假期间工资按国家规定执行。

### 3.3 事假

事假需提前申请，经部门主管审批。事假期间不计发工资。

## 第四章 费用报销

### 4.1 差旅费

出差需提前填写出差申请单，经部门主管审批。差旅费用在出差结束后一周内报销。

### 4.2 交通费

因公出行产生的交通费用，凭票据实报实销。

### 4.3 加班餐补

工作日加班超过2小时，可报销餐费补贴30元/次。

## 第五章 考勤制度

### 5.1 工作时间

标准工作时间为周一至周五 9:00-18:00，午休1小时。

### 5.2 打卡

员工需在上班和下班时各打卡一次。迟到30分钟以内扣款50元。

## 第六章 IT 管理

### 6.1 VPN 远程接入

员工可通过VPN远程接入公司内部系统，VPN账号由IT部门统一分配。

### 6.2 办公设备

公司为员工配备办公电脑和打印机等设备，员工应妥善保管。
"""

    demo_md.write_text(demo_content, encoding="utf-8")
    print(f"  演示文档: {demo_md}")
    print(f"  文件大小: {demo_md.stat().st_size} bytes")
    print(f"  章节: 6章（总则/薪资/休假/报销/考勤/IT）")

    # ── 3. 解析器配置 ───────────────────────────────────────────
    banner("3. 解析器配置")

    print(f"  当前 PDF 解析器: {settings.ingestion.parsers.get('pdf', 'docling')}")
    print(f"  当前 Markdown 解析器: {settings.ingestion.parsers.get('md', 'direct')}")
    print(f"  解析后输出目录: {settings.ingestion.parsed_doc_dir}")
    print()
    print("  可用解析器:")
    print("    docling       — 支持 pdf/docx/pptx/html (默认)")
    print("    pymupdf4llm   — 轻量 PDF 解析")
    print("    mineru        — 高精度 PDF 解析(需额外模型)")
    print("    direct        — 直接读取 md/txt")

    # ── 4. 分块策略 ─────────────────────────────────────────────
    banner("4. 分块策略")

    print(f"  当前策略: {settings.chunking.strategy}")
    print(f"  chunk_size: {settings.chunking.chunk_size}")
    print(f"  overlap:    {settings.chunking.overlap}")
    print()
    print("  三种策略对比:")
    print("    semantic      — SentenceTransformer 语义边界切分，通用")
    print("    fixed         — 固定窗口 + 滑动步长，结构弱时适用")
    print("    hierarchical  — 按 Markdown 标题层级切分，文档层级清晰时最佳")
    print()
    print("  推荐: 层级清晰的文档使用 hierarchical，通用文档使用 semantic")

    # ── 5. 执行文档处理 ─────────────────────────────────────────
    banner("5. 执行文档处理 Pipeline")

    from ingestion import create_default_pipeline

    pipeline = create_default_pipeline()
    print(f"  Pipeline 已创建: Parser → Chunker → Embedder → FAISSIndexWriter")
    print(f"  Embedding 模型: {settings.embedding.model}")

    try:
        ctx = await pipeline.run(demo_md, collection="demo_collection")

        print(f"\n  ✅ 处理完成!")
        print(f"  状态:         {ctx.status}")
        print(f"  文档 ID:      {ctx.document.doc_id if ctx.document else 'N/A'}")
        print(f"  生成分块数:   {len(ctx.chunks)}")
        print(f"  错误数:       {len(ctx.errors)}")

        if ctx.errors:
            print(f"  错误详情:")
            for err in ctx.errors:
                print(f"    - {err}")

        # 各阶段耗时
        meta = ctx.metadata
        print(f"\n  各阶段耗时:")
        print(f"    Parser:   {meta.get('parser_ms', 'N/A')} ms")
        print(f"    Chunker:  {meta.get('chunker_ms', 'N/A')} ms")
        print(f"    Embedder: {meta.get('embedder_ms', 'N/A')} ms")

        # 展示前几个 chunk
        if ctx.chunks:
            print(f"\n  Top-3 分块预览:")
            for i, chunk in enumerate(ctx.chunks[:3], 1):
                text_preview = chunk.text[:100].replace('\n', ' ')
                has_embedding = chunk.embedding is not None
                emb_dim = len(chunk.embedding) if has_embedding else 0
                print(f"    {i}. [{chunk.chunk_id}] dim={emb_dim} | {text_preview}...")

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    # ── 6. 索引文件检查 ─────────────────────────────────────────
    banner("6. FAISS 索引文件")

    idx_dir = PROJECT_ROOT / settings.faiss.index_dir / "demo_collection"
    print(f"  索引目录: {idx_dir}")

    if idx_dir.exists():
        for f in sorted(idx_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<25s} {size_kb:>8.1f} KB")
    else:
        print(f"  ⚠️ 索引目录不存在（可能使用了不同的 collection 名称）")
        # 检查 default collection
        default_idx = PROJECT_ROOT / settings.faiss.index_dir / "default"
        if default_idx.exists():
            print(f"  检查 default collection:")
            for f in sorted(default_idx.iterdir()):
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name:<25s} {size_kb:>8.1f} KB")

    # ── 7. 同 doc_id 重复写入（增量更新） ───────────────────────
    banner("7. 增量更新验证（重复写入同一 doc_id）")

    print("  再次处理同一文档（模拟增量更新）...")
    try:
        ctx2 = await pipeline.run(demo_md, collection="demo_collection")
        print(f"  ✅ 第二次处理完成, 状态: {ctx2.status}")
        print(f"  说明: FAISSIndexWriter 自动替换同 doc_id 旧向量，避免孤儿向量堆积")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # ── 清理 ────────────────────────────────────────────────────
    banner("8. 清理临时文件")

    demo_md.unlink(missing_ok=True)
    print(f"  已删除临时文档: {demo_md}")
    print(f"  索引文件保留在: {idx_dir}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 文档处理模块演示完成")
    print()
    print("  接下来可以运行检索演示:")
    print("    python examples/08_retrieval/demo_retrieval.py")
    print()
    print("  配置项 (settings.ingestion):")
    print(f"    parsed_doc_dir: {settings.ingestion.parsed_doc_dir}")
    print(f"    parsers:        {dict(settings.ingestion.parsers)}")
    print()
    print("  CLI 演示入口:")
    print("    python -m ingestion")


if __name__ == "__main__":
    asyncio.run(main())
