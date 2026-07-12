"""ParserStage — 基于 docling 的多格式文档解析"""

from ingestion.context import PipelineContext


class ParserStage:
    """使用 docling 解析 PDF/Word/Markdown → Markdown 文本"""

    name = "parser"
    fatal = True

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        source_path = ctx.document.source_path

        if not source_path.exists():
            raise FileNotFoundError(f"文件不存在: {source_path}")

        if ctx.document.file_type in ("md", "markdown"):
            # Markdown 文件直接读取，不走 docling
            ctx.document.raw_text = source_path.read_text(encoding="utf-8")
        else:
            # PDF/Word 通过 docling 解析
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(source_path))
            ctx.document.raw_text = result.document.export_to_markdown()

        ctx.document.metadata.setdefault("source_path", str(source_path))
        ctx.document.metadata.setdefault("file_size", source_path.stat().st_size)

        return ctx
