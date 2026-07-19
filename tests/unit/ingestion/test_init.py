"""create_default_pipeline 模型懒加载测试（审查 H12：并发下只加载一次）"""
import threading
import time

import sentence_transformers

import ingestion


class _SlowFakeModel:
    """模拟加载耗时的 SentenceTransformer"""

    instances = 0
    _count_lock = threading.Lock()

    def __init__(self, path):
        time.sleep(0.05)  # 放大加载窗口，暴露竞态
        with type(self)._count_lock:
            type(self).instances += 1


class TestLazyModelLoadThreadSafety:
    def test_concurrent_create_loads_model_once(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(ingestion, "_cached_embedding_model", None)
        monkeypatch.setattr(ingestion.models, "get_path", lambda t: "fake/path")
        _SlowFakeModel.instances = 0
        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", _SlowFakeModel
        )

        errors: list = []

        def build():
            try:
                ingestion.create_default_pipeline()
            except Exception as e:  # noqa: BLE001 — 测试收集任意异常
                errors.append(repr(e))

        threads = [threading.Thread(target=build) for _ in range(4)]

        # Act
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert
        assert errors == []
        assert _SlowFakeModel.instances == 1, (
            f"GB 级模型应只加载一次，实际加载 {_SlowFakeModel.instances} 次"
        )
