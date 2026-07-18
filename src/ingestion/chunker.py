"""ChunkerStage — 语义/固定/层级 三种分块策略"""

import re
import uuid

from ingestion.context import Chunk, PipelineContext, StageError


# ============================================================================
# Splitter 基类
# ============================================================================

class _BaseSplitter:
    """Splitter 基类：提供双向链表构建、token 估算等共用逻辑"""

    def _estimate_tokens(self, text: str) -> int:
        """中文字数估算 token 数（1 字 ≈ 1 token）"""
        return len(text)

    def _build_chunks(self, texts: list[str], doc_id: str = "") -> list[Chunk]:
        """为文本列表生成 Chunk，自动建立双向链表"""
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                chunk_index=i,
            )
            chunks.append(chunk)

        for i, c in enumerate(chunks):
            if i > 0:
                c.prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                c.next_chunk_id = chunks[i + 1].chunk_id

        return chunks


# ============================================================================
# FixedChunker — 固定大小 + 滑动窗口重叠
# ============================================================================

class FixedChunker(_BaseSplitter):
    """固定大小分块，相邻 chunk 有 overlap 重叠"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def splitter(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        step = max(self.chunk_size - self.overlap, 1)
        text_segments = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            text_segments.append(text[start:end])
            if end == len(text):
                break
            start += step

        return self._build_chunks(text_segments)


# ============================================================================
# HierarchicalChunker — 按标题层级分块
# ============================================================================

class HierarchicalChunker(_BaseSplitter):
    """按 Markdown 标题层级分块，保留 heading_path 到 metadata"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def splitter(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        sections = re.split(r"(?=^#{1,3}\s?)", text, flags=re.MULTILINE)

        heading_stack = []
        text_segments = []
        heading_paths = []

        for section in sections:
            if not section.strip():
                continue

            heading_match = re.match(r"^(#{1,3})\s*(.+)", section)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading_text)

            heading_path = " > ".join(heading_stack) if heading_stack else ""

            content = section
            if len(content) > self.chunk_size:
                step = max(self.chunk_size - self.overlap, 1)  # 应用 overlap 保持段内连续性
                for i in range(0, len(content), step):
                    seg = content[i: i + self.chunk_size]
                    if seg.strip():
                        text_segments.append(seg)
                        heading_paths.append(heading_path)
            else:
                text_segments.append(content)
                heading_paths.append(heading_path)

        chunks = self._build_chunks(text_segments)
        for c, hp in zip(chunks, heading_paths):
            c.metadata["heading_path"] = hp

        # 短章节间添加 overlap，保持上下文连续性
        if self.overlap > 0 and len(chunks) > 1:
            # 先保存各 chunk 原始尾部，避免原地修改导致链式偏移
            original_tails = [c.text[-self.overlap:] for c in chunks[:-1]]
            for i in range(1, len(chunks)):
                if original_tails[i - 1]:
                    chunks[i].text = original_tails[i - 1] + chunks[i].text

        return chunks


# ============================================================================
# SemanticChunker — embedding 相似度检测语义边界
# ============================================================================

class SemanticChunker(_BaseSplitter):
    """通过相邻句子 embedding 余弦相似度检测语义边界"""

    def __init__(
        self,
        embedding_model,
        chunk_size: int = 512,
        overlap: int = 64,
        threshold_percentile: float = 0.9,
        buffer_size: int = 1,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.threshold_percentile = threshold_percentile
        self.buffer_size = buffer_size

    def splitter(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        # 1. 拆分为句子
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._build_chunks([text])

        # 2. 批量计算句子 embedding
        embeddings = self.embedding_model.encode(sentences)

        # 3. 计算相邻句子余弦相似度
        import numpy as np

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(float(sim))

        # 4. 取 percentile 作为阈值
        threshold = np.percentile(similarities, self.threshold_percentile * 100)

        # 5. 标记切分点（应用 buffer 抑制邻近重复切分）
        raw_cuts = sorted(i + 1 for i, sim in enumerate(similarities) if sim < threshold)
        cut_points = set()
        last_cut = -self.buffer_size - 1
        for cut in raw_cuts:
            if cut - last_cut > self.buffer_size:
                cut_points.add(cut)
                last_cut = cut

        # 6. 按切分点合并句子
        text_segments = []
        start = 0
        for cut in sorted(cut_points):
            if cut > start:
                seg = "".join(sentences[start:cut])
                if seg.strip():
                    text_segments.append(seg)
                start = cut
        if start < len(sentences):
            seg = "".join(sentences[start:])
            if seg.strip():
                text_segments.append(seg)

        # 7. 合并过短的 segment
        text_segments = self._merge_short_segments(text_segments)

        # 8. 应用滑动窗口重叠
        chunks = self._build_chunks(text_segments)
        if self.overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].text[-self.overlap:]
                if prev_end:
                    chunks[i].text = prev_end + chunks[i].text

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """按句末标点拆分句子（不按 \\n 拆分，保留句间空白）"""
        raw = re.split(r"(?<=[。！？\.\!\?])", text)
        return [s for s in raw if s.strip()]

    def _merge_short_segments(self, segments: list[str]) -> list[str]:
        """合并过短的 segment，控制每个 segment 接近 chunk_size"""
        merged = []
        buffer = ""
        for seg in segments:
            # 超长段拆分：按 chunk_size 步进切分，不在内部做 overlap。
            # overlap 统一由外部滑动窗口处理（见 splitter() L204-208），
            # 否则子段之间会出现双重 overlap 导致文本重复。
            if self._estimate_tokens(seg) > self.chunk_size:
                if buffer.strip():
                    merged.append(buffer)
                    buffer = ""
                step = max(self.chunk_size, 1)
                for start in range(0, self._estimate_tokens(seg), step):
                    end = min(start + self.chunk_size, len(seg))
                    sub = seg[start:end]
                    if sub.strip():
                        merged.append(sub)
                continue

            if self._estimate_tokens(buffer + seg) <= self.chunk_size:
                buffer += seg
            else:
                if buffer.strip():
                    merged.append(buffer)
                buffer = seg
        if buffer.strip():
            merged.append(buffer)
        return merged


# ============================================================================
# ChunkerStage — Pipeline Stage
# ============================================================================

class ChunkerStage:
    """根据 settings.chunking.strategy 选择 splitter 并执行分块"""

    name = "chunker"
    fatal = False

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        from config import settings

        raw_text = ctx.document.raw_text
        if not raw_text or not raw_text.strip():
            ctx.errors.append(
                StageError(stage=self.name, error="empty document text, no chunks")
            )
            return ctx

        cfg = settings.chunking
        strategy = cfg.strategy

        if strategy == "semantic":
            if self.embedding_model is None:
                ctx.errors.append(
                    StageError(
                        stage=self.name,
                        error=(
                            "SemanticChunker 需要 embedding_model，"
                            "请通过 ChunkerStage(embedding_model=...) 传入"
                        ),
                        fatal=True,
                    )
                )
                return ctx
            splitter = SemanticChunker(
                embedding_model=self.embedding_model,
                chunk_size=cfg.chunk_size,
                overlap=cfg.overlap,
                threshold_percentile=cfg.semantic_threshold_percentile,
                buffer_size=cfg.semantic_buffer_size,
            )
        elif strategy == "fixed":
            splitter = FixedChunker(
                chunk_size=cfg.chunk_size,
                overlap=cfg.overlap,
            )
        elif strategy == "hierarchical":
            splitter = HierarchicalChunker(
                chunk_size=cfg.chunk_size,
                overlap=cfg.overlap,
            )
        else:
            ctx.errors.append(
                StageError(
                    stage=self.name,
                    error=f"未知分块策略: {strategy}，可选: semantic | fixed | hierarchical",
                )
            )
            return ctx

        chunks = splitter.splitter(raw_text)

        for c in chunks:
            c.doc_id = ctx.document.doc_id
            c.metadata["doc_title"] = ctx.document.title

        ctx.chunks = chunks

        if not chunks:
            ctx.errors.append(
                StageError(stage=self.name, error="chunking produced zero chunks")
            )

        return ctx
