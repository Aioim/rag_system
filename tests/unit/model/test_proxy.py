"""代理服务测试"""
import pytest
from fastapi.testclient import TestClient

from model.proxy.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestProxyHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestProxyModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "local-llm"


class TestProxyChatCompletions:
    def test_chat_completions_structure(self, client, monkeypatch):
        """验证 chat completions 响应结构（mock LLM）"""
        class _FakeLLM:
            async def ainvoke(self, prompt, **kwargs):
                return "测试回复"

        monkeypatch.setattr(
            "model.proxy.server.get_local_llm",
            lambda: _FakeLLM()
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "你好"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "测试回复"
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_with_params(self, client, monkeypatch):
        """验证参数透传"""
        captured = {}

        class _FakeLLM:
            async def ainvoke(self, prompt, **kwargs):
                captured.update(kwargs)
                return "ok"

        monkeypatch.setattr(
            "model.proxy.server.get_local_llm",
            lambda: _FakeLLM()
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0.5,
            "max_tokens": 256,
        })
        assert resp.status_code == 200
        assert captured.get("temperature") == 0.5
        assert captured.get("max_tokens") == 256

    def test_chat_completions_file_not_found(self, client, monkeypatch):
        """LLM 加载失败返回 503"""
        class _FakeLLM:
            async def ainvoke(self, prompt, **kwargs):
                raise FileNotFoundError("model.gguf not found")

        monkeypatch.setattr(
            "model.proxy.server.get_local_llm",
            lambda: _FakeLLM()
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "测试"}],
        })
        assert resp.status_code == 503
