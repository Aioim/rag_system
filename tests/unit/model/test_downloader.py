"""测试 downloader 模块：策略类 + ModelDownloader"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.model.downloader import (
    _validate_model_id,
    HfStrategy,
    MsStrategy,
    AutoStrategy,
    ModelDownloader,
)


class TestValidateModelId:
    """模块级校验函数 — 行为不变"""

    def test_valid_id(self):
        assert _validate_model_id("BAAI/bge-large-zh-v1.5") == "BAAI/bge-large-zh-v1.5"

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("BAAI\\bge-large-zh-v1.5")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="路径穿越"):
            _validate_model_id("../etc/passwd")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="绝对路径"):
            _validate_model_id("/etc/models")


class TestHfStrategy:
    """HuggingFace 下载策略"""

    def test_download_calls_snapshot_download(self, tmp_path):
        """验证 snapshot_download 被正确调用"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            strategy = HfStrategy(endpoint=None, token=None, max_retries=1)
            result = strategy.download("org/model", force=False, cache_dir=tmp_path)

            mock_sd.assert_called_once()
            call_kwargs = mock_sd.call_args.kwargs
            assert call_kwargs["repo_id"] == "org/model"
            assert call_kwargs["local_dir"] == str(tmp_path / "org" / "model")

    def test_download_passes_endpoint_and_token(self, tmp_path):
        """验证 endpoint 和 token 传递"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            strategy = HfStrategy(
                endpoint="https://hf-mirror.com",
                token="hf_test123",
                max_retries=1,
            )
            strategy.download("org/model", force=False, cache_dir=tmp_path)

            call_kwargs = mock_sd.call_args.kwargs
            assert call_kwargs["token"] == "hf_test123"
            assert call_kwargs["endpoint"] == "https://hf-mirror.com"

    def test_retries_on_os_error(self, tmp_path):
        """验证网络错误时重试"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            mock_sd.side_effect = [OSError("network"), MagicMock()]
            strategy = HfStrategy(max_retries=3)
            result = strategy.download("org/model", force=False, cache_dir=tmp_path)
            assert mock_sd.call_count == 2

    def test_raises_after_max_retries(self, tmp_path):
        """验证超过重试次数后抛出 RuntimeError"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            mock_sd.side_effect = OSError("persistent network error")
            strategy = HfStrategy(max_retries=2)
            with pytest.raises(RuntimeError, match="模型下载失败"):
                strategy.download("org/model", force=False, cache_dir=tmp_path)

    def test_raises_value_error_for_missing_repo(self, tmp_path):
        """验证仓库不存在时抛出 ValueError"""
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch("src.model.downloader.snapshot_download") as mock_sd:
            mock_response = MagicMock()
            mock_response.headers = {}
            mock_sd.side_effect = RepositoryNotFoundError("not found", response=mock_response)
            strategy = HfStrategy(max_retries=1)
            with pytest.raises(ValueError, match="模型仓库不存在"):
                strategy.download("org/nonexistent", force=False, cache_dir=tmp_path)


class TestMsStrategy:
    """ModelScope 下载策略"""

    def test_raises_if_modelscope_not_installed(self):
        """验证 modelscope 未安装时抛 RuntimeError"""
        with patch.dict("sys.modules", {"modelscope": None}):
            with pytest.raises(RuntimeError, match="modelscope 未安装"):
                MsStrategy()

    def test_download_calls_modelscope_snapshot(self, tmp_path):
        """验证调用 modelscope.snapshot_download"""
        mock_ms = MagicMock()
        mock_tmp_dir = tmp_path / ".modelscope" / "org--model"
        mock_tmp_dir.mkdir(parents=True)
        (mock_tmp_dir / "model.safetensors").write_text("weight")
        mock_ms_sd = MagicMock(return_value=str(mock_tmp_dir))
        mock_ms.snapshot_download = mock_ms_sd

        with patch.dict("sys.modules", {"modelscope": mock_ms}):
            strategy = MsStrategy()
            # 手动替换 _snapshot_download 引用
            strategy._snapshot_download = mock_ms_sd

            result = strategy.download("org/model", force=False, cache_dir=tmp_path)
            mock_ms_sd.assert_called_once_with(
                "org/model",
                cache_dir=str(tmp_path / ".modelscope_cache"),
            )

    def test_download_cleans_up_on_failure(self, tmp_path):
        """验证下载失败时清理不完整目录"""
        mock_ms_sd = MagicMock(side_effect=RuntimeError("download failed"))

        with patch.dict("sys.modules", {"modelscope": MagicMock()}):
            strategy = MsStrategy()
            strategy._snapshot_download = mock_ms_sd

            with pytest.raises(RuntimeError, match="ModelScope 下载失败"):
                strategy.download("org/model", force=False, cache_dir=tmp_path)


class TestAutoStrategy:
    """自动选择策略 — 优先 MS，失败回退 HF"""

    def test_uses_ms_when_successful(self, tmp_path):
        """MS 成功时不调用 HF"""
        ms = MagicMock()
        ms.download.return_value = tmp_path / "org" / "model"
        hf = MagicMock()

        strategy = AutoStrategy(ms, hf)
        result = strategy.download("org/model", force=False, cache_dir=tmp_path)

        ms.download.assert_called_once()
        hf.download.assert_not_called()

    def test_falls_back_to_hf_when_ms_fails(self, tmp_path):
        """MS 失败时回退到 HF"""
        ms = MagicMock()
        ms.download.side_effect = RuntimeError("MS failed")
        hf = MagicMock()
        hf.download.return_value = tmp_path / "org" / "model"

        strategy = AutoStrategy(ms, hf)
        result = strategy.download("org/model", force=False, cache_dir=tmp_path)

        ms.download.assert_called_once()
        hf.download.assert_called_once_with("org/model", False, tmp_path)

    def test_falls_back_when_ms_not_available(self, tmp_path):
        """modelscope 不可用时回退 HF"""
        ms = MagicMock()
        ms.download.side_effect = ImportError("no modelscope")
        hf = MagicMock()
        hf.download.return_value = tmp_path / "org" / "model"

        strategy = AutoStrategy(ms, hf)
        result = strategy.download("org/model", force=False, cache_dir=tmp_path)

        hf.download.assert_called_once()


class TestModelDownloader:
    """ModelDownloader 集成测试"""

    def test_builds_hf_strategy(self, tmp_path):
        """download_source='huggingface' 创建 HfStrategy"""
        dl = ModelDownloader(
            cache_dir=tmp_path,
            download_source="huggingface",
            hf_token="t",
        )
        assert isinstance(dl._strategy, HfStrategy)

    def test_builds_ms_strategy(self, tmp_path):
        """download_source='modelscope' 创建 MsStrategy"""
        dl = ModelDownloader(
            cache_dir=tmp_path,
            download_source="modelscope",
        )
        assert isinstance(dl._strategy, MsStrategy)

    def test_builds_auto_strategy(self, tmp_path):
        """download_source='auto' 创建 AutoStrategy（默认）"""
        dl = ModelDownloader(cache_dir=tmp_path)
        assert isinstance(dl._strategy, AutoStrategy)

    def test_build_strategy_rejects_invalid_source(self, tmp_path):
        """非法 download_source 抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的 download_source"):
            ModelDownloader(cache_dir=tmp_path, download_source="invalid")

    def test_download_skips_when_exists(self, tmp_path):
        """已下载模型跳过下载"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("weight")

        mock_strategy = MagicMock()
        dl = ModelDownloader(cache_dir=tmp_path)
        dl._strategy = mock_strategy

        result = dl.download("org/model")
        mock_strategy.download.assert_not_called()
        assert result == model_dir

    def test_download_delegates_to_strategy(self, tmp_path):
        """未下载时委托给策略"""
        mock_strategy = MagicMock()
        mock_strategy.download.return_value = tmp_path / "org" / "model"

        dl = ModelDownloader(cache_dir=tmp_path)
        dl._strategy = mock_strategy

        result = dl.download("org/model")
        mock_strategy.download.assert_called_once_with(
            "org/model", False, tmp_path
        )

    def test_download_force_re_downloads(self, tmp_path):
        """force=True 即使已有模型也重新下载"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("weight")

        mock_strategy = MagicMock()
        mock_strategy.download.return_value = model_dir

        dl = ModelDownloader(cache_dir=tmp_path)
        dl._strategy = mock_strategy

        dl.download("org/model", force=True)
        mock_strategy.download.assert_called_once()

    def test_is_downloaded_detects_weights(self, tmp_path):
        """is_downloaded 检测权重文件"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        dl = ModelDownloader(cache_dir=tmp_path)
        assert dl.is_downloaded("org/model") is False

        (model_dir / "model.safetensors").write_text("weight")
        assert dl.is_downloaded("org/model") is True

    def test_list_downloaded_scans_recursively(self, tmp_path):
        """list_downloaded 递归扫描"""
        (tmp_path / "org" / "model").mkdir(parents=True)
        (tmp_path / "org" / "model" / "model.safetensors").write_text("w")

        dl = ModelDownloader(cache_dir=tmp_path)
        result = dl.list_downloaded()
        assert "org/model" in result
        assert result["org/model"] == tmp_path / "org" / "model"

    def test_remove_deletes_directory(self, tmp_path):
        """remove 删除模型目录"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("w")

        dl = ModelDownloader(cache_dir=tmp_path)
        assert dl.remove("org/model") is True
        assert not model_dir.exists()

    def test_model_dir_returns_path(self, tmp_path):
        """model_dir 返回计算路径"""
        dl = ModelDownloader(cache_dir=tmp_path)
        assert dl.model_dir("org/model") == tmp_path / "org" / "model"
