"""ParserStage — 文档解析阶段，委托给可插拔的解析器后端"""

import asyncio
from pathlib import Path

from config import settings
from ingestion.context import PipelineContext
from ingestion.parsers import get_parser
from models.enums import DocumentStatus


class ParserStage:
    """文档解析 Pipeline Stage — 按配置选择解析器，委托转换并持久化为 .md 文件"""

    name = "parser"
    fatal = True

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        source_path = ctx.document.source_path

        if not source_path.exists():
            raise FileNotFoundError(f"文件不存在: {source_path}")

        # 查表获取解析器名称，未配置的格式 fallback 到 docling
        parser_name = settings.ingestion.parsers.get(
            ctx.document.file_type, "docling"
        )
        parser = get_parser(parser_name)

        # 委托解析（parse() 是同步方法，通过 to_thread 异步化）
        ctx.document.raw_text = await asyncio.to_thread(
            parser.parse, source_path, settings.ingestion.parsed_doc_dir
        )

        # 将解析后的 Markdown 写入磁盘
        md_path = self._write_markdown(ctx)
        ctx.document.metadata = {
            "source_path": str(source_path),
            "file_size": source_path.stat().st_size,
            "parsed_md_path": str(md_path),
            "parser": parser_name,
            **ctx.document.metadata,  # 已有 key 优先，保留 setdefault 语义
        }
        ctx.document.status = DocumentStatus.DONE

        return ctx

    @staticmethod
    def _write_markdown(ctx: PipelineContext) -> Path:
        """将 raw_text 写入 parsed_doc_dir / {doc_id}.md"""
        output_dir = settings.ingestion.parsed_doc_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{ctx.document.doc_id}.md"
        md_path.write_text(ctx.document.raw_text, encoding="utf-8")
        return md_path
