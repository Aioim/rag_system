"""
02_pdf_parsing.py — 文档解析：PDF 解析（含/不含图片提取）

演示内容：
  1. 配置参数说明（解析器选择 / 图片提取相关配置）
  2. 模式 A — 纯文本解析（docling 直接导出，<!-- image --> 占位符原样保留）
  3. 模式 B — 含图片解析（docling 文本 + fitz 提取图片 + 注入 ![](ref) 引用）
  4. 两种模式对比（Markdown 长度、图片引用数、耗时）

运行方式：
  cd rag0709
  python examples/11_ingestion/02_pdf_parsing.py

前置条件：
  - pip install pymupdf  （fitz 图片提取）
  - docling 已安装       （PDF → Markdown 文本）

无需 Embedding 模型或 LLM API
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples._common import PROJECT_ROOT, banner  # noqa: E402

from config import settings  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# 配置参数说明
# ═══════════════════════════════════════════════════════════════════════════
#
# settings.ingestion.parsers       — 文件扩展名 → 解析器映射
#   {pdf: docling}                  PDF 默认使用 docling
#   可选: pymupdf4llm / mineru      修改 config/{env}.yaml 切换
#
# settings.ingestion.parsed_doc_dir — 解析产物输出目录
#   默认: data/parsed_docs           图片和 Markdown 均输出到此目录
#
# settings.ingestion.mineru.device   — MinerU 设备选择（仅 mineru 解析器）
#   默认: cpu                        可选: cuda / mps
#
# settings.ingestion.mineru.models_dir — MinerU 模型目录
#   默认: local_models/mineru
#
# ── 图片提取相关（本示例） ─────────────────────────────────────────────
#
# 图片提取模式由调用方式决定，不通过全局配置切换：
#   模式 A（无图片）: 仅调用 docling export_to_markdown()，不做后处理
#   模式 B（含图片）: docling + fitz 提取图片 + 正则替换 <!-- image -->
#
# 受控参数（在代码中调整）：
#   DO_IMAGE_EXTRACTION: bool = True    # 是否执行 fitz 图片提取
#   IMAGE_FORMAT: str = "png"           # fitz 提取图片的保存格式
#
# ═══════════════════════════════════════════════════════════════════════════


# ── 演示用 PDF 路径 ──────────────────────────────────────────────────
DEMO_PDF = PROJECT_ROOT / "data/demo/OceanBase-数据库-V4.6.0-共享存储.pdf"


def _print_config_overview() -> None:
    """打印 PDF 解析相关配置参数"""
    banner("0. 配置参数说明")

    parsers = settings.ingestion.parsers
    pdf_parser = parsers.get("pdf", "docling")

    print(f"  PDF 解析器配置:")
    print(f"    settings.ingestion.parsers['pdf'] = {pdf_parser!r}")
    print(f"    切换方式: 修改 config/{{env}}.yaml → ingestion.parsers.pdf")
    print()
    print(f"  输出目录:")
    print(f"    settings.ingestion.parsed_doc_dir = {settings.ingestion.parsed_doc_dir}")
    print()
    print(f"  可用 PDF 解析器:")
    print(f"    docling       — 多格式支持，OCR/表格/图片占位符，默认推荐")
    print(f"    pymupdf4llm   — 轻量快速，适合纯文本文档")
    print(f"    mineru        — 高精度，支持 OCR + 图片提取（需额外模型）")
    print()
    print(f"  图片提取模式（本示例通过代码控制，非全局配置）:")
    print(f"    模式 A — 纯文本: docling → Markdown（<!-- image --> 占位符）")
    print(f"    模式 B — 含图片: docling + fitz 提取 → ![](ref) 实际引用")
    print(f"    切换: 在代码中设置 DO_IMAGE_EXTRACTION = True/False")
    print()
    print(f"  依赖项:")
    print(f"    docling  — PDF → Markdown（自动安装）")
    print(f"    pymupdf  — 图片提取引擎（pip install pymupdf）")


async def _parse_pdf_text_only(pdf_path: Path) -> str:
    """模式 A：纯文本解析 — docling 直接导出，不处理图片

    Returns:
        Markdown 文本，图片以 <!-- image --> 占位符保留
    """
    import time

    print(f"\n  ── 模式 A: 纯文本解析 ──")
    t0 = time.perf_counter()

    def _convert():
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()

    md_text = await asyncio.to_thread(_convert)
    elapsed = (time.perf_counter() - t0) * 1000

    placeholder_count = md_text.count("<!-- image -->")
    print(f"    耗时:        {elapsed:.0f} ms")
    print(f"    输出长度:    {len(md_text):,} 字符")
    print(f"    图片占位符:  {placeholder_count} 处（保留为 <!-- image -->）")

    # 保存
    stem = pdf_path.stem
    out_path = settings.ingestion.parsed_doc_dir / f"{stem}_text_only.md"
    out_path.write_text(md_text, encoding="utf-8")
    print(f"    产物:        {out_path}")

    return md_text


async def _parse_pdf_with_images(pdf_path: Path) -> str:
    """模式 B：含图片解析 — docling + fitz 提取图片 + 注入引用

    技术路线：
      1. docling 解析 PDF → Markdown（含 <!-- image --> 占位符）
      2. PyMuPDF (fitz) 逐页提取内嵌图片 → 保存到磁盘
      3. 正则替换 <!-- image --> → ![](images/xxx.png)
    """
    import re
    import time

    import fitz

    print(f"\n  ── 模式 B: 含图片解析 ──")

    # B1. docling 解析文本
    print(f"    [B1] docling 解析...")
    t0 = time.perf_counter()

    def _convert():
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()

    md_text = await asyncio.to_thread(_convert)
    text_ms = (time.perf_counter() - t0) * 1000
    placeholder_count = md_text.count("<!-- image -->")
    print(f"      耗时: {text_ms:.0f} ms, {len(md_text):,} 字符, {placeholder_count} 个占位符")

    # B2. fitz 提取图片
    print(f"    [B2] PyMuPDF 提取图片...")
    t0 = time.perf_counter()

    stem = pdf_path.stem
    out_dir = settings.ingestion.parsed_doc_dir / f"{stem}_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(str(pdf_path))
    total_pages = len(pdf_doc)
    extracted = []

    for page_num in range(total_pages):
        for img_idx, img_info in enumerate(pdf_doc[page_num].get_images()):
            xref = img_info[0]
            try:
                base_image = pdf_doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                filename = f"page{page_num + 1:02d}_img{img_idx + 1:02d}.{ext}"
                (out_dir / filename).write_bytes(img_bytes)
                extracted.append({
                    "filename": filename,
                    "page": page_num + 1,
                    "size": len(img_bytes),
                    "width": base_image["width"],
                    "height": base_image["height"],
                })
            except Exception:
                pass

    pdf_doc.close()
    fitz_ms = (time.perf_counter() - t0) * 1000
    print(f"      耗时: {fitz_ms:.0f} ms, 提取 {len(extracted)} 张 → {out_dir}")

    # B3. 注入图片引用
    print(f"    [B3] 注入图片引用...")
    t0 = time.perf_counter()

    refs = [
        f"![{img['filename']}]({stem}_images/{img['filename']})"
        for img in extracted
    ]

    def _replace(match, refs=refs, idx=[0]):
        if idx[0] < len(refs):
            idx[0] += 1
            return refs[idx[0] - 1]
        return "<!-- image (无对应) -->"

    md_with_images = re.sub(r"<!-- image -->", _replace, md_text)
    inject_ms = (time.perf_counter() - t0) * 1000

    # 保存
    out_path = settings.ingestion.parsed_doc_dir / f"{stem}_with_images.md"
    out_path.write_text(md_with_images, encoding="utf-8")
    ref_count = md_with_images.count("![")
    print(f"      耗时: {inject_ms:.0f} ms, 注入 {ref_count} 个图片引用")
    print(f"    总耗时:  {text_ms + fitz_ms + inject_ms:.0f} ms")
    print(f"    产物:    {out_path}")

    # 图片统计
    if extracted:
        total_kb = sum(i["size"] for i in extracted) / 1024
        print(f"    图片:    {len(extracted)} 张, {total_kb:.0f} KB, "
              f"尺寸 {min(i['width'] for i in extracted)}×{min(i['height'] for i in extracted)}"
              f" ~ {max(i['width'] for i in extracted)}×{max(i['height'] for i in extracted)}")

    return md_with_images


async def main():
    # ── 0. 配置参数说明 ──────────────────────────────────────────
    _print_config_overview()

    # ── 检查 PDF 文件 ────────────────────────────────────────────
    banner("检查 PDF 文件")
    if not DEMO_PDF.exists():
        print(f"  ⚠️ PDF 文件不存在: {DEMO_PDF}")
        print(f"  请将 PDF 文件放置到该路径后重试")
        return

    print(f"  ✅ {DEMO_PDF.name}")
    print(f"     大小: {DEMO_PDF.stat().st_size / 1024:.0f} KB")

    # ── 1. 模式 A — 纯文本解析 ───────────────────────────────────
    banner("1. 模式 A — 纯文本解析（无图片）")
    md_text_only = await _parse_pdf_text_only(DEMO_PDF)

    # ── 2. 模式 B — 含图片解析 ───────────────────────────────────
    banner("2. 模式 B — 含图片解析")
    md_with_images = await _parse_pdf_with_images(DEMO_PDF)

    # ── 3. 两种模式对比 ──────────────────────────────────────────
    banner("3. 两种模式对比")

    text_img_count = md_text_only.count("<!-- image -->")
    real_img_count = md_with_images.count("![")

    print(f"  {'指标':<20s} {'模式A (纯文本)':<20s} {'模式B (含图片)':<20s}")
    print(f"  {'-'*60}")
    print(f"  {'Markdown 长度':<20s} {len(md_text_only):<20,} {len(md_with_images):<20,}")
    print(f"  {'图片引用':<20s} {f'{text_img_count} 个占位符':<20s} {f'{real_img_count} 个实际引用':<20s}")
    print(f"  {'产物文件':<20s} {'*_text_only.md':<20s} {'*_with_images.md':<20s}")
    print(f"  {'图片目录':<20s} {'(无)':<20s} {'*_images/':<20s}")
    print()
    print(f"  适用场景:")
    print(f"    模式 A — 检索场景（向量化不需要图片）、纯文本分析")
    print(f"    模式 B — 文档预览、知识展示、需要保留图表信息")

    # ── 4. 预览 ──────────────────────────────────────────────────
    banner("4. 模式 B 产物预览（前 20 行）")

    lines = md_with_images.split("\n")[:20]
    for line in lines:
        if len(line) > 120:
            line = line[:117] + "..."
        print(f"    {line}")
    if len(md_with_images.split("\n")) > 20:
        print(f"    ... (共 {len(md_with_images.split(chr(10)))} 行)")

    banner("✅ PDF 解析演示完成")
    print()
    print("  产物位置:")
    print(f"    纯文本: {settings.ingestion.parsed_doc_dir / (DEMO_PDF.stem + '_text_only.md')}")
    print(f"    含图片: {settings.ingestion.parsed_doc_dir / (DEMO_PDF.stem + '_with_images.md')}")
    print(f"    图片:   {settings.ingestion.parsed_doc_dir / (DEMO_PDF.stem + '_images')}/")
    print()
    print("  下一步: 03_chunking_strategies.py — 三种分块策略对比")


if __name__ == "__main__":
    asyncio.run(main())
