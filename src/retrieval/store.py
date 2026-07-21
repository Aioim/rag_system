"""FAISSStore — FAISS 索引 + docstore 只读访问（每 collection 一个实例）"""
import json
import threading

import faiss
import numpy as np

from logger import logger
from models.chunk import Chunk


class FAISSStore:
    """单 collection 的 FAISS 索引 + docstore 只读封装

    线程安全懒加载；reload() 热重载并递增 version（BM25 缓存失效依据）。

    使用 _state 元组原子快照 (_index, _docstore, _id_map)，
    保证跨字段一致性：reload() 原子替换整个元组，读取者
    通过单次引用读取获得一致的三元组快照。
    """

    def __init__(self, collection: str):
        # 防路径遍历：collection 将拼接到索引目录路径，拒绝分隔符 / .. / 空值
        if not collection or "/" in collection or "\\" in collection or ".." in collection:
            raise ValueError(f"非法 collection 名称: {collection!r}")
        self.collection = collection
        self.version = 0
        self._lock = threading.Lock()
        self._state: tuple = (None, {}, {})  # (_index, _docstore, _id_map)
        self._loaded = False

    # ---- 加载 ----------------------------------------------------------

    def load(self) -> None:
        """幂等加载；collection 目录不存在抛 ValueError"""
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self._load_unlocked()

    def _load_unlocked(self) -> None:
        from config import settings

        index_dir = settings.faiss.index_dir / self.collection
        if not index_dir.exists():
            raise ValueError(f"Collection '{self.collection}' 不存在: {index_dir}")

        index_path = index_dir / "index.faiss"
        index = None
        if index_path.exists():
            try:
                index = faiss.read_index(str(index_path))
            except Exception as e:
                raise RuntimeError(
                    f"Collection '{self.collection}' FAISS 索引加载失败: {e}"
                ) from e
            if isinstance(index, faiss.IndexIVFFlat):
                index.nprobe = settings.faiss.nprobe
                try:
                    index.make_direct_map()   # MMR 需 reconstruct 原始向量
                except Exception as e:
                    raise RuntimeError(
                        f"Collection '{self.collection}' FAISS direct_map 构建失败: {e}"
                    ) from e

        docstore = {}
        docstore_path = index_dir / "docstore.json"
        if docstore_path.exists():
            try:
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # 显式失败：静默降级为空 docstore 会导致检索返回全空结果且无错误信号；
                # 不置 _loaded，文件修复后下次调用可重试
                raise RuntimeError(
                    f"Collection '{self.collection}' docstore 加载失败: {e}"
                ) from e

        id_map = {
            entry["faiss_id"]: cid
            for cid, entry in docstore.items()
            if "faiss_id" in entry
        }

        # 原子替换整个状态元组，保证跨字段一致性
        self._state = (index, docstore, id_map)
        self._loaded = True

    def reload(self) -> None:
        """热重载（索引更新后调用）；version 递增使 BM25 缓存失效"""
        with self._lock:
            self._loaded = False
            self._load_unlocked()
            self.version += 1

    # ---- 查询 ----------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        index = self._state[0]  # 通过 _state 元组原子读取
        return index is None or index.ntotal == 0

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """docstore entry → 新 Chunk 实例（每次新建，防调用方污染）"""
        _, docstore, _ = self._state  # 原子快照
        entry = docstore.get(chunk_id)
        if entry is None:
            return None
        return Chunk(
            chunk_id=chunk_id,
            doc_id=entry.get("doc_id", ""),
            text=entry.get("text", ""),
            chunk_index=entry.get("chunk_index", 0),
            prev_chunk_id=entry.get("prev_chunk_id"),
            next_chunk_id=entry.get("next_chunk_id"),
            metadata=dict(entry.get("metadata", {})),
        )

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        """FAISS 向量搜索 → 按相关度降序的 chunk_id 列表"""
        if k <= 0:
            return []
        index, _, id_map = self._state  # 原子快照，保证 index 与 id_map 同代
        if index is None or index.ntotal == 0:
            return []
        k = min(k, index.ntotal)
        _, ids = index.search(
            vector.reshape(1, -1).astype(np.float32), k
        )
        result = []
        for faiss_id in ids[0]:
            if faiss_id < 0:
                continue
            chunk_id = id_map.get(int(faiss_id))
            if chunk_id is None:
                logger.warning(
                    "FAISS id %s 在 docstore 中不存在，已跳过", faiss_id
                )
                continue
            result.append(chunk_id)
        return result

    def reconstruct(self, chunk_id: str) -> np.ndarray | None:
        """取 chunk 的原始向量（MMR 多样性计算用）"""
        index, docstore, _ = self._state  # 原子快照，保证 index 与 docstore 同代
        entry = docstore.get(chunk_id)
        if entry is None or "faiss_id" not in entry or index is None:
            return None
        try:
            return index.reconstruct(int(entry["faiss_id"]))
        except (RuntimeError, ValueError, TypeError):
            # RuntimeError: faiss 内部错误；ValueError/TypeError: faiss_id 数据损坏
            return None

    def all_chunks(self) -> list[tuple[str, str]]:
        """(chunk_id, text) 列表，供 BM25 建索引"""
        _, docstore, _ = self._state  # 原子快照
        return [
            (cid, entry.get("text", ""))
            for cid, entry in docstore.items()
        ]


# ---- 模块级 store 缓存 ---------------------------------------------------

_stores: dict[str, FAISSStore] = {}
_stores_lock = threading.Lock()


def get_store(collection: str) -> FAISSStore:
    """获取（并懒加载）collection 对应的 FAISSStore 单例"""
    store = _stores.get(collection)
    if store is None:
        with _stores_lock:
            store = _stores.get(collection)
            if store is None:
                store = FAISSStore(collection)
                _stores[collection] = store
    store.load()
    return store


def reset_stores() -> None:
    """清空 store 缓存（测试用）"""
    with _stores_lock:
        _stores.clear()
