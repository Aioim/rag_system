"""
01_document_parsing.py — 文档解析：解析器选择与执行

演示内容：
  1. 解析器配置概览（各文件格式对应的解析器）
  2. 创建真实感 Markdown 文档
  3. 执行解析（ParserStage）
  4. 解析产物检查（落盘文件 + 文本统计）
  5. PDF 解析 + 图片提取（docling 文本 + fitz 图片 → 含图 Markdown）

运行方式：
  cd rag0709
  python examples/11_ingestion/01_document_parsing.py

无需 Embedding 模型或 LLM API
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples._common import PROJECT_ROOT, banner  # noqa: E402

from config import settings  # noqa: E402


# ── 演示用文档（真实感员工手册，含层级标题 + 表格 + 列表）──────────
DEMO_DOC = """# 员工手册

## 第一章 总则

本手册适用于公司全体员工，旨在规范公司管理，保障员工权益。公司秉持"以人为本、诚信经营"的理念，为员工提供良好的工作环境和发展平台。

## 第二章 薪资福利

### 2.1 薪资结构

员工薪资由以下三部分组成：

| 组成部分 | 占比 | 说明 |
|---------|------|------|
| 基本工资 | 60% | 根据职级确定 |
| 岗位工资 | 25% | 根据岗位价值评估 |
| 绩效工资 | 15% | 根据季度绩效考核浮动 |

每月 **15 日** 发放上月工资，遇节假日顺延。

### 2.2 五险一金

公司为所有正式员工缴纳以下社会保险和住房公积金：

- 养老保险（单位 16%，个人 8%）
- 医疗保险（单位 8%，个人 2%）
- 失业保险（单位 0.5%，个人 0.5%）
- 工伤保险（单位 0.2%，个人 0%）
- 生育保险（单位 0.8%，个人 0%）
- 住房公积金（单位 12%，个人 12%）

### 2.3 年终奖金

根据公司年度经营业绩和个人绩效考核结果，于每年 **1 月** 发放上年度年终奖金。考核等级与奖金系数对照：

| 考核等级 | 奖金系数 |
|---------|---------|
| S (卓越) | 3.0× 月薪 |
| A (优秀) | 2.0× 月薪 |
| B (良好) | 1.5× 月薪 |
| C (合格) | 1.0× 月薪 |
| D (待改进) | 0.5× 月薪 |

## 第三章 休假制度

### 3.1 带薪年休假

- 工作满 1 年不满 10 年：每年 **5 天**
- 工作满 10 年不满 20 年：每年 **10 天**
- 工作满 20 年以上：每年 **15 天**

年假可累积至次年 **3 月 31 日**，逾期未休视为自动放弃。

### 3.2 病假

员工因病需要休息的，应持 **二级甲等以上医院** 出具的病假证明。病假期间工资按以下标准执行：

- 病假 1-2 天：不扣工资
- 病假 3-30 天：按基本工资的 80% 发放
- 病假超过 30 天：按国家有关规定执行

### 3.3 事假

事假需 **提前 1 个工作日** 申请，经部门主管审批。事假期间不计发工资。每月事假累计不得超过 3 天。

## 第四章 费用报销

### 4.1 差旅费

出差需提前填写 **出差申请单**（OA 系统 → 行政管理 → 出差申请），经部门主管审批。差旅费用在出差结束后 **5 个工作日内** 报销。

### 4.2 交通费

因公出行产生的交通费用，凭票据实报实销。市内交通优先使用公共交通工具。

### 4.3 加班餐补

工作日加班超过 **2 小时**，可报销餐费补贴 **30 元/次**；周末加班超过 **4 小时**，可报销餐费补贴 **50 元/次**。

## 第五章 考勤制度

### 5.1 工作时间

标准工作时间为 **周一至周五 9:00-18:00**，午休 **12:00-13:00**（1 小时）。

弹性工作制：可申请 8:00-17:00 或 10:00-19:00，需经部门主管审批。

### 5.2 考勤规则

- 员工需在上班和下班时各打卡一次（使用企业微信或门禁卡）
- 迟到 **30 分钟以内**：扣款 50 元
- 迟到超过 **30 分钟**：按旷工半天处理
- 忘记打卡：需在 **2 个工作日内** 提交补卡申请

## 第六章 IT 管理

### 6.1 VPN 远程接入

员工可通过 VPN 远程接入公司内部系统：

- VPN 客户端下载：[it.example.com/vpn](http://it.example.com/vpn)
- 账号：员工工号
- 初始密码：身份证后 6 位（首次登录需修改）
- 支持平台：Windows / macOS / iOS / Android

### 6.2 办公设备

公司为员工配备以下办公设备：

| 设备 | 配置 | 更换周期 |
|------|------|---------|
| 笔记本电脑 | ThinkPad X1 / MacBook Pro | 3 年 |
| 显示器 | 27" 4K | 5 年 |
| 键盘鼠标 | 罗技 MX 系列 | 2 年 |

员工离职时需归还所有办公设备，如有损坏按折旧赔偿。
"""


def _demo_pdf_parsing(pdf_path):
    """解析真实 PDF 文件并提取图片，生成含图片引用的 Markdown

    技术路线：
      1. docling 解析 PDF → Markdown 文本（含 <!-- image --> 占位符）
      2. PyMuPDF (fitz) 提取内嵌图片 → 保存为 PNG
      3. 将 <!-- image --> 替换为 ![](images/xxx.png) 引用
    """
    import re
    import time

    import fitz

    from docling.document_converter import DocumentConverter

    print(f"  PDF 文件: {pdf_path.name}")
    print(f"  文件大小: {pdf_path.stat().st_size / 1024:.0f} KB")

    # ── 5a. docling 解析文本 ──────────────────────────────────────
    print(f"\n  [5a] docling 解析文本...")
    t0 = time.perf_counter()
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    md_text = result.document.export_to_markdown()
    docling_ms = (time.perf_counter() - t0) * 1000
    print(f"    耗时: {docling_ms:.0f} ms")
    print(f"    Markdown 长度: {len(md_text):,} 字符")
    # 统计 <!-- image --> 占位符数量
    placeholder_count = md_text.count("<!-- image -->")
    print(f"    图片占位符: {placeholder_count} 处")

    # ── 5b. fitz 提取图片 ─────────────────────────────────────────
    print(f"\n  [5b] PyMuPDF 提取图片...")
    t0 = time.perf_counter()

    # 输出目录
    pdf_stem = pdf_path.stem
    out_dir = settings.ingestion.parsed_doc_dir / f"{pdf_stem}_pdf_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(str(pdf_path))
    total_pages = len(pdf_doc)
    extracted_images = []

    for page_num in range(total_pages):
        page = pdf_doc[page_num]
        images = page.get_images()
        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            try:
                base_image = pdf_doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                # 保存图片
                img_filename = f"page{page_num + 1:02d}_img{img_idx + 1:02d}.{ext}"
                img_path = out_dir / img_filename
                img_path.write_bytes(img_bytes)
                extracted_images.append({
                    "filename": img_filename,
                    "page": page_num + 1,
                    "size": len(img_bytes),
                    "width": base_image["width"],
                    "height": base_image["height"],
                })
            except Exception as e:
                print(f"    ⚠️ 提取失败 page={page_num + 1} img={img_idx}: {e}")

    pdf_doc.close()
    fitz_ms = (time.perf_counter() - t0) * 1000
    print(f"    耗时: {fitz_ms:.0f} ms")
    print(f"    提取图片: {len(extracted_images)} 张 → {out_dir}")

    # ── 5c. 注入图片引用 ──────────────────────────────────────────
    print(f"\n  [5c] 注入图片引用...")

    # 策略：将 <!-- image --> 占位符依次替换为实际图片引用
    image_refs = [
        f"![{img['filename']}]({pdf_stem}_pdf_images/{img['filename']})"
        for img in extracted_images
    ]

    # 替换 <!-- image --> 为实际图片引用，多余的占位符保留原样
    def _replace_image_placeholder(match, refs=image_refs, idx_counter=[0]):
        idx = idx_counter[0]
        if idx < len(refs):
            idx_counter[0] += 1
            return refs[idx]
        return "<!-- image (无对应提取图片) -->"

    md_with_images = re.sub(r"<!-- image -->", _replace_image_placeholder, md_text)

    # 保存最终 Markdown
    final_md_path = settings.ingestion.parsed_doc_dir / f"{pdf_stem}_with_images.md"
    final_md_path.write_text(md_with_images, encoding="utf-8")
    print(f"    最终 Markdown: {final_md_path}")
    print(f"    图片引用数: {md_with_images.count('![')} 处")

    # ── 5d. 展示统计与预览 ────────────────────────────────────────
    print(f"\n  [5d] 统计与预览")
    print(f"    PDF 页数:       {total_pages}")
    print(f"    文档长度:       {len(md_text):,} 字符")
    print(f"    提取图片:       {len(extracted_images)} 张")
    if extracted_images:
        total_img_kb = sum(img["size"] for img in extracted_images) / 1024
        print(f"    图片总大小:     {total_img_kb:.0f} KB")
        print(f"    图片尺寸范围:   {min(i['width'] for i in extracted_images)}×{min(i['height'] for i in extracted_images)} ~ {max(i['width'] for i in extracted_images)}×{max(i['height'] for i in extracted_images)}")
        print(f"\n    前 3 张图片:")
        for img in extracted_images[:3]:
            print(f"      {img['filename']}  page={img['page']}  {img['width']}×{img['height']}  {img['size']:,} bytes")

    # 展示最终 Markdown 前 800 字符
    print(f"\n    最终 Markdown 预览（前 800 字符）:")
    print(f"    {'─' * 56}")
    for line in md_with_images.split("\n")[:25]:
        # 截断过长的图片引用显示
        if len(line) > 120:
            line = line[:117] + "..."
        print(f"    {line}")
    if len(md_with_images.split("\n")) > 25:
        print(f"    ... (共 {len(md_with_images.split(chr(10)))} 行)")


async def main():
    # ── 1. 解析器配置概览 ─────────────────────────────────────────
    banner("1. 解析器配置概览")

    parsers = settings.ingestion.parsers
    print(f"  文件格式 → 解析器映射:")
    for ext, parser_name in sorted(parsers.items()):
        print(f"    .{ext:<8s} → {parser_name}")

    print(f"\n  解析产物输出目录: {settings.ingestion.parsed_doc_dir}")

    print(f"\n  可用解析器说明:")
    print(f"    docling      — 支持 pdf/docx/pptx/html (默认推荐)")
    print(f"    pymupdf4llm  — 轻量 PDF 解析")
    print(f"    mineru       — 高精度 PDF 解析 (需额外模型)")
    print(f"    direct       — 直接读取 md/txt (无外部依赖)")

    # ── 2. 创建演示文档 ───────────────────────────────────────────
    banner("2. 创建演示文档")

    tmp_dir = Path(tempfile.mkdtemp())
    demo_file = tmp_dir / "员工手册.md"
    demo_file.write_text(DEMO_DOC, encoding="utf-8")

    print(f"  文件路径: {demo_file}")
    print(f"  文件大小: {demo_file.stat().st_size:,} bytes")
    print(f"  章节数:   6 章（总则/薪资/休假/报销/考勤/IT）")

    # ── 3. 执行解析 ──────────────────────────────────────────────
    banner("3. 执行解析 (ParserStage)")

    import time
    from ingestion.context import Document, PipelineContext
    from ingestion.parser import ParserStage

    doc = Document(
        doc_id="demo_parsing_001",
        source_path=demo_file,
        file_type=demo_file.suffix.lstrip("."),
        title=demo_file.stem,
        collection="demo_parsing",
    )
    ctx = PipelineContext(document=doc, status="running")

    stage = ParserStage()
    t0 = time.perf_counter()
    try:
        ctx = await stage.run(ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  解析器:    {ctx.document.metadata.get('parser', 'unknown')}")
        print(f"  耗时:      {elapsed_ms:.0f} ms")
        print(f"  状态:      {ctx.status}")
        print(f"  文档长度:  {len(ctx.document.raw_text):,} 字符")
        print(f"  段落数:    {len(ctx.document.raw_text.split(chr(10)+chr(10)))} 段")
    except Exception as e:
        print(f"  ❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── 4. 解析产物检查 ───────────────────────────────────────────
    banner("4. 解析产物检查")

    md_path = Path(ctx.document.metadata.get("parsed_md_path", ""))
    if md_path.exists():
        size = md_path.stat().st_size
        print(f"  落盘文件: {md_path}")
        print(f"  文件大小: {size:,} bytes")
        # 展示内容摘要
        lines = ctx.document.raw_text.split("\n")
        headings = [l.strip() for l in lines if l.startswith("#")]
        print(f"  标题层级: {len(headings)} 个标题")
        for h in headings:
            level = h.count("#") if not h.startswith(" ") else 0
            indent = "  " * (level - 1)
            print(f"    {indent}{h}")
    else:
        print(f"  ⚠️ 解析产物文件不存在: {md_path}")

    # ── 5. PDF 解析（含图片提取）───────────────────────────────────
    banner("5. PDF 解析（含图片提取）")

    pdf_path = PROJECT_ROOT / "data/demo/OceanBase-数据库-V4.6.0-共享存储.pdf"
    if not pdf_path.exists():
        print(f"  ⚠️ PDF 文件不存在: {pdf_path}")
        print(f"  跳过 PDF 解析演示")
    else:
        _demo_pdf_parsing(pdf_path)

    # ── 6. 清理与总结 ─────────────────────────────────────────────
    banner("6. 清理临时文件")

    demo_file.unlink(missing_ok=True)
    print(f"  已删除临时文档: {demo_file}")
    print(f"  解析产物保留在: {settings.ingestion.parsed_doc_dir}")
    print(f"  可手动查看: {md_path}")

    banner("✅ 文档解析演示完成")
    print()
    print("  下一步: 02_chunking_strategies.py — 三种分块策略对比")


if __name__ == "__main__":
    asyncio.run(main())
