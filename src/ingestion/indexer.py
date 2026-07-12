"""FAISSIndexWriter — FAISS 索引持久化"""

import json
from pathlib import Path

import faiss
import numpy as np

from logger import logger


def _sanitize_metadata(meta: dict) -> dict:
    """递归清洗 metadata，将非 JSON 可序列化值转为字符串"""
    import numpy as np

    def _sanitize(value):
        if value is None or isinstance(value, (bool, str, int, float)):
            return value
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_sanitize(v) for v in value]
        return str(value)

    return _sanitize(meta)


class FAISSIndexWriter:
    """将带 embedding 的 chunks 写入 FAISS 索引 + docstore"""

    def write(self, chunks: list, collection: str) -> None:
        if not chunks:
            return

        from config import settings

        cfg = settings.faiss
        expected_dim = cfg["dimension"]

        # 维度校验
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} 无 embedding，无法写入索引")
            if len(c.embedding) != expected_dim:
                raise ValueError(
                    f"Chunk {c.chunk_id} embedding 维度 {len(c.embedding)} "
                    f"与配置 {expected_dim} 不一致"
                )

        # 索引目录
        index_dir = Path(cfg["index_dir"]) / collection
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "index.faiss"
        docstore_path = index_dir / "docstore.json"

        # 加载已有 docstore
        existing_docstore = {}
        if docstore_path.exists():
            with open(docstore_path, encoding="utf-8") as f:
                existing_docstore = json.load(f)

        # 构建向量矩阵
        vectors = np.array([c.embedding for c in chunks], dtype=np.float32)

        # 加载或创建 FAISS 索引
        is_cosine = cfg["metric_type"] == "COSINE"
        _METRIC_IP = getattr(faiss, "METRIC_INNER_PRODUCT", 10)
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            actual_type = type(index).__name__.upper()
            config_type = cfg["index_type"].upper().replace("_", "")
            if config_type not in actual_type:
                logger.warning(
                    "磁盘索引类型 %s 与配置 index_type=%s 不一致，使用磁盘索引",
                    actual_type, cfg["index_type"],
                )
        else:
            dim = expected_dim
            if cfg["index_type"] == "IVF_FLAT":
                if is_cosine:
                    quantizer = faiss.IndexFlatIP(dim)
                    index = faiss.IndexIVFFlat(
                        quantizer, dim, cfg["nlist"], _METRIC_IP
                    )
                else:
                    quantizer = faiss.IndexFlatL2(dim)
                    index = faiss.IndexIVFFlat(
                        quantizer, dim, cfg["nlist"], faiss.METRIC_L2
                    )
            elif cfg["index_type"] == "FLAT":
                if is_cosine:
                    index = faiss.IndexFlatIP(dim)
                else:
                    index = faiss.IndexFlatL2(dim)
            else:
                index = faiss.IndexFlatIP(dim)

        # 训练 IVF
        if isinstance(index, faiss.IndexIVFFlat) and not index.is_trained:
            if len(vectors) >= cfg["nlist"]:
                index.train(vectors)
            else:
                # 向量不足时降级为 Flat，保留 metric 语义
                if is_cosine:
                    index = faiss.IndexFlatIP(expected_dim)
                else:
                    index = faiss.IndexFlatL2(expected_dim)

        # COSINE: normalize
        if is_cosine:
            faiss.normalize_L2(vectors)

        # 添加向量
        start_id = index.ntotal
        index.add(vectors)

        # 持久化索引
        faiss.write_index(index, str(index_path))

        # 持久化 docstore
        new_entries = {}
        for i, c in enumerate(chunks):
            entry = {
                "faiss_id": start_id + i,
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
            }
            if c.prev_chunk_id:
                entry["prev_chunk_id"] = c.prev_chunk_id
            if c.next_chunk_id:
                entry["next_chunk_id"] = c.next_chunk_id
            if c.metadata:
                entry["metadata"] = _sanitize_metadata(c.metadata)
            new_entries[c.chunk_id] = entry

        existing_docstore.update(new_entries)
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump(existing_docstore, f, ensure_ascii=False, indent=2)
