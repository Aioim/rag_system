"""
02_pdf_parsing.py — 文档解析：PDF 解析（含/不含图片提取）

演示内容：
  1. 配置参数说明（解析器选择 / 图片提取相关配置）
  2. 统一解析方法 parse_pdf() — 通过 extract_images 参数控制模式
  3. 两种模式对比（Markdown 长度、图片引用数、产物文件）

使用的 src/ingestion 封装方法：
  - ingestion.parsers.get_parser()  — 按名称获取解析器实例（带缓存）
  - BaseParser.parse()              — 同步解析，通过 asyncio.to_thread 异步化

运行方式：
  cd rag0709
  python examples/11_ingestion/02_pdf_parsing.py

前置条件：
  - pip install pymupdf  （fitz 图片提取，仅 extract_images=True 时需要）
  - docling 已安装       （PDF → Markdown 文本，src/ingestion 自动加载）

无需 Embedding 模型或 LLM API
"""

import asyncio
import re
import sys
import time
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
# ── 图片提取控制 ───────────────────────────────────────────────────────
#
# parse_pdf(pdf_path, extract_images=True/False)
#   extract_images=False — 纯文本模式：docling → Markdown，<!-- image --> 占位
#                          产物: {stem}_text_only.md
#   extract_images=True  — 含图片模式：docling + fitz 提取 → ![](ref) 引用
#                          产物: {stem}_with_images.md + {stem}_images/
#
# ═══════════════════════════════════════════════════════════════════════════


DEMO_PDF = PROJECT_ROOT / "data/demo/OceanBase-数据库-V4.6.0-共享存储.pdf"


def _print_config_overview() -> None:
    """打印 PDF 解析相关配置参数"""
    banner("0. 配置参数说明")

    pdf_parser = settings.ingestion.parsers.get("pdf", "docling")

    print(f"  PDF 解析器配置:")
    print(f"    settings.ingestion.parsers['pdf'] = {pdf_parser!r}")
    print(f"    切换方式: 修改 config/{{env}}.yaml → ingestion.parsers.pdf")
    print()
    print(f"  输出目录:")
    print(f"    settings.ingestion.parsed_doc_dir = {settings.ingestion.parsed_doc_dir}")
    print()
    print(f"  parse_pdf() 参数:")
    print(f"    extract_images=False  → 纯文本（<!-- image --> 占位符）")
    print(f"    extract_images=True   → 含图片（fitz 提取 + ![](ref) 引用）")
    print()
    print(f"  可用 PDF 解析器:")
    print(f"    docling       — 多格式支持，OCR/表格/图片占位符，默认推荐")
    print(f"    pymupdf4llm   — 轻量快速，适合纯文本文档")
    print(f"    mineru        — 高精度，支持 OCR + 图片提取（需额外模型）")
    print()
    print(f"  依赖项:")
    print(f"    docling  — PDF → Markdown（自动安装）")
    print(f"    pymupdf  — 图片提取引擎（pip install pymupdf，仅 extract_images=True）")


async def parse_pdf(pdf_path: Path, *, extract_images: bool = False) -> str:
    """统一的 PDF 解析方法，通过 extract_images 控制是否提取图片

    内部使用 src/ingestion 封装的 get_parser() 获取解析器实例，
    文本解析步骤两种模式共享同一逻辑。

    Args:
        pdf_path: PDF 源文件路径
        extract_images: False=纯文本（<!-- image -->占位），True=提取图片+注入引用

    Returns:
        Markdown 文本
    """
    from ingestion.parsers import get_parser

    stem = pdf_path.stem
    out_dir = settings.ingestion.parsed_doc_dir

    # ── 1. 文本解析（两种模式共享，使用 src/ingestion 封装）────────
    label = "含图片" if extract_images else "纯文本"
    print(f"\n  ── parse_pdf(extract_images={extract_images}) {label} ──")

    parser = get_parser("docling")
    print(f"    解析器: {parser.name} ({type(parser).__name__})")
    print(f"    API:    ingestion.parsers.get_parser('docling')")

    t0 = time.perf_counter()
    md_text = await asyncio.to_thread(parser.parse, pdf_path, out_dir)
    text_ms = (time.perf_counter() - t0) * 1000

    placeholder_count = md_text.count("<!-- image -->")
    print(f"    文本解析: {text_ms:.0f} ms, {len(md_text):,} 字符, {placeholder_count} 个占位符")

    # ── 2. 保存 / 图片提取（分支点）────────────────────────────────
    if not extract_images:
        out_path = out_dir / f"{stem}_text_only.md"
        out_path.write_text(md_text, encoding="utf-8")
        print(f"    产物:     {out_path}")
        return md_text

    # ── 2b. fitz 提取图片（src/ingestion 暂无封装）─────────────────
    import fitz

    print(f"    fitz 提取: ", end="")
    t0 = time.perf_counter()

    img_dir = out_dir / f"{stem}_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(str(pdf_path))
    extracted = []

    for page_num in range(len(pdf_doc)):
        for img_idx, img_info in enumerate(pdf_doc[page_num].get_images()):
            try:
                base = pdf_doc.extract_image(img_info[0])
                filename = f"page{page_num + 1:02d}_img{img_idx + 1:02d}.{base['ext']}"
                (img_dir / filename).write_bytes(base["image"])
                extracted.append({
                    "filename": filename,
                    "size": len(base["image"]),
                    "width": base["width"],
                    "height": base["height"],
                })
            except Exception:
                pass
    pdf_doc.close()

    fitz_ms = (time.perf_counter() - t0) * 1000
    print(f"{fitz_ms:.0f} ms, {len(extracted)} 张 → {img_dir}")

    # ── 2c. 注入图片引用 ──────────────────────────────────────────
    refs = [f"![{i['filename']}]({stem}_images/{i['filename']})" for i in extracted]

    def _replace(match, refs=refs, idx=[0]):
        if idx[0] < len(refs):
            idx[0] += 1
            return refs[idx[0] - 1]
        return "<!-- image (无对应) -->"

    md_text = re.sub(r"<!-- image -->", _replace, md_text)

    out_path = out_dir / f"{stem}_with_images.md"
    out_path.write_text(md_text, encoding="utf-8")

    total_kb = sum(i["size"] for i in extracted) / 1024 if extracted else 0
    print(f"    图片引用: {md_text.count('![')} 处")
    print(f"    总耗时:   {text_ms + fitz_ms:.0f} ms")
    print(f"    产物:     {out_path}")
    if extracted:
        print(f"    图片:     {len(extracted)} 张, {total_kb:.0f} KB")

    return md_text


async def main():
    # ── 0. 配置参数说明 ──────────────────────────────────────────
    _print_config_overview()

    # ── 检查 PDF 文件 ────────────────────────────────────────────
    banner("检查 PDF 文件")
    if not DEMO_PDF.exists():
        print(f"  ⚠️ PDF 文件不存在: {DEMO_PDF}")
        return

    print(f"  ✅ {DEMO_PDF.name}")
    print(f"     大小: {DEMO_PDF.stat().st_size / 1024:.0f} KB")

    # ── 1. 纯文本模式 ────────────────────────────────────────────
    banner("1. extract_images=False — 纯文本（无图片目录）")
    md_text_only = await parse_pdf(DEMO_PDF, extract_images=False)

    # ── 2. 含图片模式 ────────────────────────────────────────────
    banner("2. extract_images=True — 含图片")
    md_with_images = await parse_pdf(DEMO_PDF, extract_images=True)

    # ── 3. 对比 ──────────────────────────────────────────────────
    banner("3. 两种模式对比")

    text_placeholders = f"{md_text_only.count('<!--')} 个占位符"
    real_refs = f"{md_with_images.count('![')} 个实际引用"
    print(f"  {'指标':<20s} {'extract_images=False':<24s} {'extract_images=True':<24s}")
    print(f"  {'-'*68}")
    print(f"  {'Markdown 长度':<20s} {len(md_text_only):<24,} {len(md_with_images):<24,}")
    print(f"  {'图片处理':<20s} {text_placeholders:<24s} {real_refs:<24s}")
    print(f"  {'产物文件':<20s} {'*_text_only.md':<24s} {'*_with_images.md':<24s}")
    print(f"  {'图片目录':<20s} {'(不创建)':<24s} {'*_images/ (创建)':<24s}")
    print()
    print(f"  适用场景:")
    print(f"    extract_images=False — 文档入库、检索预处理、纯文本分析")
    print(f"    extract_images=True  — 文档预览、知识展示、保留图表信息")

    # ── 4. 预览 ──────────────────────────────────────────────────
    banner("4. 含图片模式产物预览（前 20 行）")

    for line in md_with_images.split("\n")[:20]:
        print(f"    {line[:117] + '...' if len(line) > 120 else line}")
    print(f"    ... (共 {len(md_with_images.split(chr(10)))} 行)")

    banner("✅ PDF 解析演示完成")


if __name__ == "__main__":
    asyncio.run(main())
