"""使用 MinerU 解析 PDF 并输出 Markdown"""
import os
from pathlib import Path

os.environ.setdefault("ENV", "dev")

from config import settings
from ingestion.parsers import get_parser, reset_parser_cache

pdf_path = Path(r"C:\Users\AIO\Desktop\OceanBase-数据库-V4.6.0-共享存储.pdf")
output_path = Path(r"C:\Users\AIO\Desktop\OceanBase-数据库-V4.6.0-共享存储_mineru.md")

out_dir = settings.ingestion.parsed_doc_dir
out_dir.mkdir(parents=True, exist_ok=True)

print(f"源文件: {pdf_path}  (大小: {pdf_path.stat().st_size:,} bytes)")

reset_parser_cache()
parser = get_parser("mineru")
print("正在解析（53页，预计 2-5 分钟）...")
md_text = parser.parse(pdf_path, output_dir=out_dir)

output_path.write_text(md_text, encoding="utf-8")

print(f"\n=== 解析完成 ===")
print(f"输出: {output_path}")
print(f"字符: {len(md_text):,}  行: {md_text.count(chr(10)):,}")
