"""FAISSIndexWriter — FAISS 索引持久化"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import faiss
import numpy as np

from logger import logger

if TYPE_CHECKING:
    from ingestion.context import Chunk


def _sanitize_metadata(meta: dict) -> dict:
    """递归清洗 metadata，将非 JSON 可序列化值转为字符串。

    带深度限制和循环检测，防止 RecursionError。
    """
    _max_depth = 50

    def _sanitize(value, _depth: int = 0, _seen: set[int] | None = None):
        if _depth > _max_depth:
            return str(value)
        if value is None or isinstance(value, (bool, str, int, float)):
            return value
        # numpy 类型优先转换（np.bool_ 不是 bool 子类，需单独处理）
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        # 循环引用检测
        if isinstance(value, (dict, list, tuple)):
            if _seen is None:
                _seen = set()
            obj_id = id(value)
            if obj_id in _seen:
                return str(value)
            _seen.add(obj_id)
        if isinstance(value, dict):
            return {k: _sanitize(v, _depth + 1, _seen) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_sanitize(v, _depth + 1, _seen) for v in value]
        return str(value)

    return _sanitize(meta)


def _create_index(dim: int, is_cosine: bool) -> faiss.Index:
    """按 metric 类型创建 FAISS 索引"""
    if is_cosine:
        return faiss.IndexFlatIP(dim)
    return faiss.IndexFlatL2(dim)


class FAISSIndexWriter:
    """将带 embedding 的 chunks 写入 FAISS 索引 + docstore"""

    @staticmethod
    def _create_new_index(cfg, dim: int, is_cosine: bool) -> faiss.Index:
        """按配置创建空索引（IVF_FLAT 或 Flat）"""
        _metric_ip = getattr(faiss, "METRIC_INNER_PRODUCT", 0)
        if cfg.index_type == "IVF_FLAT":
            if is_cosine:
                quantizer: faiss.IndexFlat = faiss.IndexFlatIP(dim)
                return faiss.IndexIVFFlat(quantizer, dim, cfg.nlist, _metric_ip)
            quantizer = faiss.IndexFlatL2(dim)
            return faiss.IndexIVFFlat(quantizer, dim, cfg.nlist, faiss.METRIC_L2)
        return _create_index(dim, is_cosine)

    def _rebuild_without_stale(
        self,
        index: faiss.Index,
        docstore: dict,
        stale_ids: set[str],
        cfg,
        dim: int,
        is_cosine: bool,
    ) -> tuple[dict, faiss.Index]:
        """重建索引剔除 stale 条目的向量，保留条目按原顺序重映射 faiss_id"""
        kept_items = sorted(
            ((cid, e) for cid, e in docstore.items() if cid not in stale_ids),
            key=lambda kv: kv[1]["faiss_id"],
        )

        new_index = self._create_new_index(cfg, dim, is_cosine)
        if kept_items:
            if isinstance(index, faiss.IndexIVFFlat):
                index.make_direct_map()
            kept_vectors = np.vstack(
                [index.reconstruct(int(e["faiss_id"])) for _, e in kept_items]
            ).astype(np.float32)
            # 已存储向量在写入时归一化过（COSINE），无需再归一化
            if isinstance(new_index, faiss.IndexIVFFlat) and not new_index.is_trained:
                if len(kept_vectors) >= cfg.nlist:
                    new_index.train(kept_vectors)
                else:
                    new_index = _create_index(dim, is_cosine)
            new_index.add(kept_vectors)

        new_docstore = {
            cid: {**e, "faiss_id": new_id}
            for new_id, (cid, e) in enumerate(kept_items)
        }
        return new_docstore, new_index

    def write(self, chunks: list[Chunk], collection: str) -> None:
        if not chunks:
            return

        # 防路径遍历：与 retrieval/store.py 保持一致
        if not collection or "/" in collection or "\\" in collection or ".." in collection:
            raise ValueError(f"非法 collection 名称: {collection!r}")

        from config import settings

        cfg = settings.faiss
        expected_dim = cfg.dimension

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
        index_dir = cfg.index_dir / collection
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
        is_cosine = cfg.metric_type == "COSINE"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            actual_type = type(index).__name__.upper()
            config_type = cfg.index_type.upper().replace("_", "")
            if not actual_type.endswith(config_type):
                logger.warning(
                    "磁盘索引类型 %s 与配置 index_type=%s 不一致，使用磁盘索引",
                    actual_type, cfg.index_type,
                )
        else:
            index = self._create_new_index(cfg, expected_dim, is_cosine)

        # 增量去重：同 doc_id 重复写入时重建索引剔除旧向量，避免孤儿向量堆积
        incoming_doc_ids = {c.doc_id for c in chunks}
        stale_ids = {
            cid for cid, e in existing_docstore.items()
            if e.get("doc_id") in incoming_doc_ids
        }
        if stale_ids:
            existing_docstore, index = self._rebuild_without_stale(
                index, existing_docstore, stale_ids, cfg, expected_dim, is_cosine
            )

        # COSINE: 训练前归一化，确保 IVF 质心与存储向量在同一空间
        if is_cosine:
            faiss.normalize_L2(vectors)

        # 训练 IVF
        if isinstance(index, faiss.IndexIVFFlat) and not index.is_trained:
            if len(vectors) >= cfg.nlist:
                index.train(vectors)
            else:
                # 向量不足时降级为 Flat，保留 metric 语义
                index = _create_index(expected_dim, is_cosine)

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
            json.dump(existing_docstore, f, ensure_ascii=False)
