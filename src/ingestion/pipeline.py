"""IngestionPipeline — Stage 编排器"""

import time
import uuid
from pathlib import Path
from typing import Protocol

from ingestion.context import Document, PipelineContext, StageError
from logger import logger


class IndexWriter(Protocol):
    """FAISSIndexWriter 协议（避免循环依赖，在 indexer.py 中实现）"""

    def write(self, chunks: list, collection: str) -> None: ...


class IngestionPipeline:
    """离线文档处理 Pipeline 编排器

    依次执行 stages，记录耗时和状态，最后调用 index_writer 持久化。
    """

    def __init__(self, stages: list, index_writer: IndexWriter):
        self.stages = stages
        self.index_writer = index_writer

    async def run(
        self, file_path: Path, collection: str = "default"
    ) -> PipelineContext:
        # 1. 构造 Document
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source_path=file_path,
            file_type=file_path.suffix.lstrip(".").lower(),
            title=file_path.stem,
            collection=collection,
        )
        ctx = PipelineContext(document=doc, status="running")

        # 2. 遍历 stages
        for stage in self.stages:
            stage_name = getattr(stage, "name", "unknown")
            stage_fatal = getattr(stage, "fatal", True)
            ctx.current_stage = stage_name
            t0 = time.perf_counter()
            try:
                ctx = await stage.run(ctx)
            except Exception as e:
                ctx.errors.append(
                    StageError(stage=stage_name, error=str(e), fatal=stage_fatal)
                )
                logger.error("Stage [%s] 失败: %s", stage_name, e)
                if stage_fatal:
                    ctx.status = "failed"
                    return ctx
            finally:
                ctx.metadata[f"{stage_name}_ms"] = (
                    time.perf_counter() - t0
                ) * 1000

        # 3. 写入索引
        try:
            self.index_writer.write(ctx.chunks, collection)
        except Exception as e:
            ctx.errors.append(
                StageError(stage="index_writer", error=str(e), fatal=True)
            )
            ctx.status = "failed"
            return ctx

        # 4. 完成（非 fatal 错误不影响 status）
        has_fatal = any(e.fatal for e in ctx.errors)
        ctx.status = "failed" if has_fatal else "done"
        return ctx
