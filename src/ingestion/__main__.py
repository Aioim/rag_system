"""Ingestion Pipeline 演示入口 — python -m ingestion"""

import asyncio
import sys
import tempfile
from pathlib import Path


DEMO_MARKDOWN = """\
# 企业知识库 — 员工手册

## 第一章 入职流程

新员工入职时应携带以下材料到人力资源部报到：
1. 身份证原件及复印件
2. 学历证书复印件
3. 一寸免冠照片 2 张

人力资源部将在 3 个工作日内完成入职手续办理。

## 第二章 考勤制度

公司实行弹性工作制，核心工作时间为 10:00-16:00。
员工每日工作时长不少于 8 小时。迟到 30 分钟以上需向直属领导报备。

月度全勤奖励 500 元，季度全勤额外奖励 1000 元。

## 第三章 报销流程

差旅报销需在返回后 5 个工作日内提交报销申请，
附上所有票据原件。审批流程：直属领导 → 部门负责人 → 财务审核。

单笔报销金额超过 5000 元需提前申请预审批。

## 第四章 技术架构

公司内部系统采用微服务架构，主要技术栈包括：
- 后端: Python FastAPI + LangChain + LangGraph
- 前端: React 18 + TypeScript + Ant Design
- 数据库: PostgreSQL 15 + Redis 7 + Milvus 2.4
- 部署: Docker + Kubernetes

## 第五章 安全规范

所有生产环境密码必须使用强密码策略（12 位以上，含大小写字母+数字+特殊字符）。
访问生产服务器需通过堡垒机，禁止直接 SSH 连接。
敏感数据（客户信息、财务数据）须经 Fernet 加密存储。
"""


async def run_demo():
    print("=" * 60)
    print("  Ingestion Pipeline 演示")
    print("=" * 60)

    # 1. 创建临时 Markdown 文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(DEMO_MARKDOWN)
        tmp_path = Path(f.name)

    try:
        print(f"\n📄 测试文档: {tmp_path.name}")
        print(f"   大小: {tmp_path.stat().st_size} bytes\n")

        # 2. 检查模型状态
        from model import models

        print("🔍 检查 Embedding 模型...")
        status = models.status()
        if not status.get("embedding"):
            print("⚠️  Embedding 模型未下载，尝试下载...")
            try:
                models.download("embedding")
                print("✅ 下载完成")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                print("   请确保: 1) 设置 HUGGINGFACE_TOKEN 环境变量")
                print("          2) 网络可访问 HuggingFace 或镜像站")
                sys.exit(1)

        # 3. 创建 Pipeline
        from ingestion import create_default_pipeline

        print("🏗️  创建 Ingestion Pipeline...")
        pipeline = create_default_pipeline()
        print("   Pipeline 创建成功\n")

        # 4. 运行 Pipeline
        print("🚀 开始处理文档...")
        print("-" * 40)

        ctx = await pipeline.run(tmp_path, collection="demo")

        print("-" * 40)
        print(f"\n📊 处理结果:")
        print(f"   状态: {ctx.status}")
        print(f"   文档: {ctx.document.title}")

        # 阶段耗时
        stages_ms = {
            k.replace("_ms", ""): v
            for k, v in ctx.metadata.items()
            if k.endswith("_ms")
        }
        for stage_name, ms in stages_ms.items():
            print(f"   {stage_name}: {ms:.1f}ms")

        # 分块统计
        if ctx.chunks:
            print(f"\n📦 分块统计:")
            print(f"   总数: {len(ctx.chunks)}")
            sizes = [len(c.text) for c in ctx.chunks]
            print(f"   平均大小: {sum(sizes) / len(sizes):.0f} 字符")
            print(f"   最小/最大: {min(sizes)}/{max(sizes)} 字符")
            embedded = sum(1 for c in ctx.chunks if c.embedding is not None)
            print(f"   已 embedding: {embedded}/{len(ctx.chunks)}")

        # 错误报告
        if ctx.errors:
            print(f"\n⚠️  错误 ({len(ctx.errors)}):")
            for err in ctx.errors:
                marker = "[FATAL]" if err.fatal else "[WARN] "
                print(f"   {marker} {err.stage}: {err.error}")

        # 5. 验证索引
        from config import settings

        index_dir = Path(settings.faiss["index_dir"]) / "demo"
        if index_dir.exists():
            index_size = sum(
                f.stat().st_size for f in index_dir.rglob("*") if f.is_file()
            )
            print(f"\n📁 索引已写入: {index_dir}")
            print(f"   文件大小: {index_size:,} bytes")

            import json

            docstore_path = index_dir / "docstore.json"
            if docstore_path.exists():
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)
                print(f"   docstore 条目: {len(docstore)}")

        print(f"\n{'=' * 60}")
        if ctx.status == "done":
            print("  ✅ Pipeline 演示成功完成")
        else:
            print(f"  ⚠️  Pipeline 完成 (状态: {ctx.status})")
        print(f"{'=' * 60}")

    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    """CLI 入口 — rag-ingest"""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
