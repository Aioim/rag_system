"""IngestionPipeline — Stage 编排器"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Protocol

from ingestion.context import Chunk, Document, PipelineContext, StageError
from ingestion.stage import Stage
from logger import logger


class IndexWriter(Protocol):
    """FAISSIndexWriter 协议（避免循环依赖，在 indexer.py 中实现）"""

    def write(self, chunks: list[Chunk], collection: str) -> None: ...


class IngestionPipeline:
    """离线文档处理 Pipeline 编排器

    依次执行 stages，记录耗时和状态，最后调用 index_writer 持久化。
    """

    def __init__(self, stages: list[Stage], index_writer: IndexWriter):
        self.stages = stages
        self.index_writer = index_writer

    async def run(
        self, file_path: Path, collection: str = "default"
    ) -> PipelineContext:
        # 1. 构造 Document（doc_id 由文件绝对路径哈希确定性生成：
        #    同一文件重复入库得到相同 doc_id，indexer 才能识别并替换旧向量）
        doc_id = hashlib.sha256(
            str(file_path.resolve()).encode("utf-8")
        ).hexdigest()[:32]
        doc = Document(
            doc_id=doc_id,
            source_path=file_path,
            file_type=file_path.suffix.lstrip(".").lower(),
            title=file_path.stem,
            collection=collection,
        )
        ctx = PipelineContext(document=doc, status="running")

        # 2. 遍历 stages
        for stage in self.stages:
            stage_name = getattr(stage, "name", "unknown")
            stage_fatal = getattr(stage, "fatal", None)
            if stage_fatal is None:
                stage_fatal = True  # 未知阶段默认视为 fatal
            ctx.current_stage = stage_name
            t0 = time.perf_counter()
            try:
                result = await stage.run(ctx)
                if result is None:
                    raise RuntimeError(
                        f"Stage [{stage_name}] 返回了 None，"
                        "所有 Stage 必须返回 PipelineContext"
                    )
                ctx = result
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

        # 3. 写入索引（to_thread 避免 FAISS I/O 阻塞事件循环）
        try:
            await asyncio.to_thread(
                self.index_writer.write, ctx.chunks, collection
            )
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
