"""
04_full_pipeline.py — 文档处理：完整 Pipeline 与 FAISS 索引

演示内容：
  1. 依赖检查（Embedding 模型自动下载）
  2. 完整处理 Pipeline（Parser → Chunker → Embedder → FAISSIndexWriter）
  3. 各阶段耗时与分块统计
  4. FAISS 索引文件验证
  5. 增量更新验证（重复写入同一 doc_id，无孤儿向量）
  6. 检索验证（端到端确认索引入库可检索）

运行方式：
  cd rag0709
  python examples/11_ingestion/04_full_pipeline.py

前置条件：Embedding 模型已下载（脚本会自动检查并下载）
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples._common import PROJECT_ROOT, banner, check_embedding_model  # noqa: E402

from config import settings  # noqa: E402


# ── 演示用文档 ─────────────────────────────────────────────────────
DEMO_DOC = """# 员工手册

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


async def main():
    # ── 1. 依赖检查 ──────────────────────────────────────────────
    banner("1. 检查依赖")
    if not check_embedding_model(auto_download=True):
        print("\n  ❌ Embedding 模型下载失败，无法继续。")
        print("  请检查 HUGGINGFACE_TOKEN 环境变量和网络连接后重试。")
        return

    # ── 2. 创建演示文档 ───────────────────────────────────────────
    banner("2. 创建演示文档")

    tmp_dir = Path(tempfile.mkdtemp())
    demo_file = tmp_dir / "员工手册.md"
    demo_file.write_text(DEMO_DOC, encoding="utf-8")
    print(f"  文件: {demo_file}")
    print(f"  大小: {demo_file.stat().st_size:,} bytes")

    # ── 3. 执行完整 Pipeline ──────────────────────────────────────
    banner("3. 执行完整 Pipeline")

    from ingestion import create_default_pipeline

    pipeline = create_default_pipeline()
    print(f"  Pipeline: Parser → Chunker({settings.chunking.strategy}) → Embedder → FAISSIndexWriter")
    print(f"  Embedding: {settings.embedding.model}")

    try:
        ctx = await pipeline.run(demo_file, collection="demo_ingestion")
    except Exception as e:
        print(f"  ❌ Pipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        demo_file.unlink(missing_ok=True)
        return

    print(f"\n  ✅ 处理完成")
    print(f"  状态:       {ctx.status}")
    print(f"  文档 ID:    {ctx.document.doc_id if ctx.document else 'N/A'}")
    print(f"  分块数:     {len(ctx.chunks)}")
    print(f"  错误数:     {len(ctx.errors)}")

    if ctx.errors:
        for e in ctx.errors:
            print(f"    ⚠️ {e.stage}: {e.error}")

    # 各阶段耗时
    meta = ctx.metadata
    print(f"\n  各阶段耗时:")
    for stage_name in ["parser", "chunker", "embedder"]:
        ms = meta.get(f"{stage_name}_ms", "N/A")
        if isinstance(ms, (int, float)):
            print(f"    {stage_name:<10s} {ms:.0f} ms")
        else:
            print(f"    {stage_name:<10s} {ms}")

    # 分块预览
    if ctx.chunks:
        print(f"\n  Top-3 分块预览:")
        for i, chunk in enumerate(ctx.chunks[:3], 1):
            text_preview = chunk.text[:100].replace("\n", " ")
            has_emb = chunk.embedding is not None
            dim = len(chunk.embedding) if has_emb else 0
            print(f"    {i}. [{chunk.chunk_id[:8]}...] dim={dim} | {text_preview}...")

    # ── 4. FAISS 索引验证 ─────────────────────────────────────────
    banner("4. FAISS 索引验证")

    idx_dir = PROJECT_ROOT / settings.faiss.index_dir / "demo_ingestion"
    print(f"  索引目录: {idx_dir}")

    if idx_dir.exists():
        for f in sorted(idx_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<25s} {size_kb:>8.1f} KB")

        # 检查 docstore 条目数
        import json
        docstore_path = idx_dir / "docstore.json"
        if docstore_path.exists():
            with open(docstore_path, encoding="utf-8") as f:
                docstore = json.load(f)
            print(f"    docstore 条目数: {len(docstore)}")
    else:
        print(f"  ⚠️ 索引目录不存在")

    # ── 5. 增量更新验证 ───────────────────────────────────────────
    banner("5. 增量更新验证（同 doc_id 重复写入）")

    print("  再次处理同一文档...")
    try:
        ctx2 = await pipeline.run(demo_file, collection="demo_ingestion")
        print(f"  ✅ 第二次处理完成, 状态: {ctx2.status}")
        print(f"  说明: FAISSIndexWriter 自动替换同 doc_id 旧向量，避免孤儿向量堆积")

        # 检查 docstore 条目数不变
        docstore_path = idx_dir / "docstore.json"
        if docstore_path.exists():
            import json
            with open(docstore_path, encoding="utf-8") as f:
                docstore = json.load(f)
            print(f"  docstore 条目数: {len(docstore)} (应与首次一致)")
    except Exception as e:
        print(f"  ❌ 增量更新失败: {e}")

    # ── 6. 检索验证 ──────────────────────────────────────────────
    banner("6. 检索验证（端到端）")

    try:
        from retrieval import get_retrieval_layer
        from models.context import PipelineContext as OnlineContext

        online_ctx = OnlineContext(
            query="年假有多少天？",
            rewritten_queries=["年假有多少天？"],
            collection="demo_ingestion",
        )

        retrieval_layer = get_retrieval_layer()
        online_ctx = await retrieval_layer.retrieve(online_ctx)

        if online_ctx.candidates:
            print(f"  ✅ 检索成功")
            print(f"  粗召回候选: {len(online_ctx.candidates)} 条")
            if online_ctx.reranked:
                print(f"  精排结果:   {len(online_ctx.reranked)} 条")
                top = online_ctx.reranked[0]
                preview = top.text[:100].replace("\n", " ")
                print(f"  Top-1 得分: {top.rerank_score:.4f} | {preview}...")
        else:
            print(f"  ⚠️ 无检索结果（索引可能为空）")
    except Exception as e:
        print(f"  ⚠️ 检索验证失败: {e}")

    # ── 7. 清理与总结 ─────────────────────────────────────────────
    demo_file.unlink(missing_ok=True)
    print(f"\n  临时文档已删除: {demo_file}")
    print(f"  索引文件保留在: {idx_dir}")

    banner("✅ 文档处理 Pipeline 演示完成")
    print()
    print("  接下来可运行检索演示:")
    print("    python examples/08_retrieval/02_retrieve_demo.py")


if __name__ == "__main__":
    asyncio.run(main())
