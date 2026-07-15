"""retrieval 测试共享 fixture — 临时小维度 FAISS 环境"""
import pytest

from config import settings
from ingestion.context import Chunk
from ingestion.indexer import FAISSIndexWriter

DIM = 8


def one_hot(i: int, scale: float = 1.0) -> list[float]:
    v = [0.0] * DIM
    v[i] = scale
    return v


def make_chunk(
    i: int,
    text: str,
    vec: list[float],
    doc_id: str = "d1",
    prev_id: str | None = None,
    next_id: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        doc_id=doc_id,
        text=text,
        chunk_index=i,
        prev_chunk_id=prev_id,
        next_chunk_id=next_id,
        embedding=vec,
    )


def write_chunks(chunks: list[Chunk], collection: str = "test") -> None:
    FAISSIndexWriter().write(chunks, collection)


@pytest.fixture
def faiss_env(tmp_path):
    """tmp 索引目录 + 8 维 + FLAT 索引；结束后还原配置并清空 store 缓存"""
    cfg = settings.faiss
    saved = (cfg.index_dir, cfg.dimension, cfg.index_type, cfg.nlist)
    cfg.index_dir = tmp_path
    cfg.dimension = DIM
    cfg.index_type = "FLAT"
    cfg.nlist = 4
    from retrieval.store import reset_stores

    reset_stores()
    yield tmp_path
    cfg.index_dir, cfg.dimension, cfg.index_type, cfg.nlist = saved
    reset_stores()
