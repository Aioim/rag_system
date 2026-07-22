"""model 推理接口单元测试"""
import threading
import time

import numpy as np
import pytest

from model import models
from model import inference


# ============================================================================
# Fake 模型
# ============================================================================

class _FakeEncoder:
    """模拟 SentenceTransformer，统计实例化次数"""

    instances = 0
    _lock = threading.Lock()

    def __init__(self, path_or_model_name, device="cpu", **kwargs):
        time.sleep(0.05)
        with type(self)._lock:
            type(self).instances += 1
        self._path = str(path_or_model_name)
        self._device = device
        self._kwargs = kwargs

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)
        return np.array([[0.1, 0.2, 0.3]] * len(texts), dtype=np.float32)


class _FakeCrossEncoder:
    """模拟 CrossEncoder，统计实例化次数"""

    instances = 0
    _lock = threading.Lock()

    def __init__(self, path_or_model_name, device="cpu", **kwargs):
        with type(self)._lock:
            type(self).instances += 1
        self._path = str(path_or_model_name)
        self._device = device
        self._kwargs = kwargs

    def rank(self, query, documents, **kwargs):
        return [
            {"corpus_id": i, "score": 1.0 - i * 0.1}
            for i in range(len(documents))
        ]


# ============================================================================
# fixture
# ============================================================================

@pytest.fixture(autouse=True)
def reset_inference_cache(monkeypatch):
    """每个测试前重置推理缓存，确保测试隔离"""
    inference._reset_cache()
    monkeypatch.setattr(models, "get_path", lambda t: "/fake/model/path")
    monkeypatch.setattr(models, "_initialized", True, raising=False)
    yield
    inference._reset_cache()


# ============================================================================
# encode 测试
# ============================================================================

class TestEncode:
    def test_encode_single_text(self, monkeypatch):
        """单条文本返回 1D ndarray"""
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)
        result = models.encode("你好世界")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_encode_multiple_texts(self, monkeypatch):
        """多条文本返回 2D ndarray"""
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)
        texts = ["文本一", "文本二", "文本三"]
        result = models.encode(texts)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 3)

    def test_encode_kwargs_passthrough(self, monkeypatch):
        """**kwargs 透传给底层 model.encode()"""
        received_kwargs = {}

        class _KwargsRecorder:
            def encode(self, texts, **kwargs):
                received_kwargs.update(kwargs)
                n = len(texts) if isinstance(texts, list) else 1
                return np.array([[0.1]] * n)

        monkeypatch.setattr(inference, "SentenceTransformer", _KwargsRecorder)
        models.encode(["文本"], batch_size=16, normalize_embeddings=True)
        assert received_kwargs.get("batch_size") == 16
        assert received_kwargs.get("normalize_embeddings") is True

    def test_encode_model_not_downloaded_raises(self, monkeypatch):
        """模型未下载时抛 RuntimeError"""
        monkeypatch.setattr(models, "get_path", lambda t: None)
        with pytest.raises(RuntimeError, match="模型未下载|Embedding 模型未下载"):
            models.encode("测试")


# ============================================================================
# rerank 测试
# ============================================================================

class TestRerank:
    def test_rerank_returns_ranked_list(self, monkeypatch):
        """返回 list[dict]，每项含 corpus_id + score"""
        monkeypatch.setattr(inference, "CrossEncoder", _FakeCrossEncoder)
        result = models.rerank("查询", ["文档A", "文档B", "文档C"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert "corpus_id" in result[0]
        assert "score" in result[0]
        assert result[0]["score"] > result[-1]["score"]

    def test_rerank_kwargs_passthrough(self, monkeypatch):
        """**kwargs 透传给底层 model.rank()"""
        received_kwargs = {}

        class _KwargsRecorder:
            def rank(self, query, documents, **kwargs):
                received_kwargs.update(kwargs)
                return [{"corpus_id": 0, "score": 1.0}]

        monkeypatch.setattr(inference, "CrossEncoder", _KwargsRecorder)
        models.rerank("查询", ["文档"], top_k=10, return_documents=True)
        assert received_kwargs.get("top_k") == 10
        assert received_kwargs.get("return_documents") is True

    def test_rerank_model_not_downloaded_raises(self, monkeypatch):
        """模型未下载时抛 RuntimeError"""
        monkeypatch.setattr(models, "get_path", lambda t: None)
        with pytest.raises(RuntimeError, match="模型未下载|Rerank 模型未下载"):
            models.rerank("查询", ["文档"])


# ============================================================================
# generate 测试
# ============================================================================

class TestGenerate:
    def test_generate_raises_not_implemented(self):
        """generate() 抛 NotImplementedError，消息含方案说明"""
        with pytest.raises(NotImplementedError) as exc_info:
            models.generate("什么是最佳方案？")
        msg = str(exc_info.value)
        assert "llama-cpp-python" in msg
        assert "GGUF" in msg


# ============================================================================
# 缓存 & 线程安全测试
# ============================================================================

class TestModelCaching:
    def test_model_instance_cached(self, monkeypatch):
        """连续两次访问返回同一实例"""
        _FakeEncoder.instances = 0
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)

        m1 = models.embedding_model
        m2 = models.embedding_model

        assert m1 is m2
        assert _FakeEncoder.instances == 1

    def test_cross_encoder_instance_cached(self, monkeypatch):
        """CrossEncoder 连续访问也复用"""
        _FakeCrossEncoder.instances = 0
        monkeypatch.setattr(inference, "CrossEncoder", _FakeCrossEncoder)

        ce1 = models.cross_encoder
        ce2 = models.cross_encoder

        assert ce1 is ce2
        assert _FakeCrossEncoder.instances == 1

    def test_concurrent_load_once(self, monkeypatch):
        """4 线程并发首次访问只加载一次"""
        _FakeEncoder.instances = 0
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)

        errors: list = []

        def load_model():
            try:
                _ = models.embedding_model
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=load_model) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert _FakeEncoder.instances == 1, (
            f"模型应只加载一次，实际加载 {_FakeEncoder.instances} 次"
        )
