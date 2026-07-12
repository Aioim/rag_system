"""ParserStage — 基于 docling 的多格式文档解析"""

import threading

from ingestion.context import PipelineContext


class ParserStage:
    """使用 docling 解析 PDF/Word/Markdown → Markdown 文本"""

    name = "parser"
    fatal = True

    _converter = None  # 延迟加载，跨文档复用
    _converter_lock = threading.Lock()

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        source_path = ctx.document.source_path

        if not source_path.exists():
            raise FileNotFoundError(f"文件不存在: {source_path}")

        if ctx.document.file_type in ("md", "markdown"):
            ctx.document.raw_text = source_path.read_text(encoding="utf-8")
        else:
            with ParserStage._converter_lock:
                if ParserStage._converter is None:
                    from docling.document_converter import DocumentConverter

                    ParserStage._converter = DocumentConverter()

                result = ParserStage._converter.convert(str(source_path))
            ctx.document.raw_text = result.document.export_to_markdown()

        ctx.document.metadata.setdefault("source_path", str(source_path))
        ctx.document.metadata.setdefault("file_size", source_path.stat().st_size)

        return ctx
