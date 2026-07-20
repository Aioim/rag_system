# Model Downloader 重构 — 集成魔搭下载

**日期**: 2026-07-20
**状态**: 已确认
**关联**: `src/model/downloader.py`

## 背景

当前 `ModelDownloader` 只支持 HuggingFace/hf-mirror 下载，魔搭下载通过独立函数 `download_from_modelscope()` 实现，存在以下问题：

1. **代码重复**：两个下载路径各自实现了 `_validate_model_id`、已下载检查、权重文件检测
2. **API 割裂**：`ModelManager` 只使用 `ModelDownloader`，无法利用魔搭国内直连优势
3. **不可配置**：没有配置项让用户选择下载源

## 目标

将魔搭下载集成到 `ModelDownloader` 类中，通过全局配置统一选择下载源。消除代码重复，保持公共接口不变。

## 设计

### 架构：策略模式

```
                    ┌─────────────────────┐
                    │   ModelDownloader   │
                    │  (public API 不变)   │
                    └─────────┬───────────┘
                              │ 委托 download()
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌───▼────────┐ ┌───▼──────────┐
     │ HfStrategy      │ │ MsStrategy │ │ AutoStrategy │
     │ (huggingface)   │ │ (modelscope)│ │ (MS→HF fallback)
     └─────────────────┘ └────────────┘ └──────────────┘
```

- `ModelDownloader` 保持现有公共 API 不变
- 只有 `download()` 行为委托给策略
- `is_downloaded` / `list_downloaded` / `remove` / `model_dir` 为本地文件操作，所有策略共享，不进入协议
- `AutoStrategy` 组合 `MsStrategy` + `HfStrategy`，先 MS 后 HF

### 策略协议

```python
class DownloadStrategy(Protocol):
    """下载策略协议 — 只覆盖下载行为"""
    def download(self, model_id: str, force: bool,
                 cache_dir: Path, **kwargs) -> Path: ...
```

策略自身无状态（依赖作为构造参数传入），方法签名统一。

### 三个策略

| 策略 | 说明 | 构造参数 |
|------|------|----------|
| `HfStrategy` | 现有 `snapshot_download` 逻辑 + 指数退避重试 | `endpoint`, `token`, `max_retries` |
| `MsStrategy` | 现有 `modelscope.snapshot_download` 逻辑 | 无 |
| `AutoStrategy` | 组合 Hf + Ms，先 MS 后 HF fallback | `ms: MsStrategy`, `hf: HfStrategy` |

### 配置

`ModelConfig` 新增字段：

```python
download_source: str = "auto"  # "huggingface" | "modelscope" | "auto"
```

`config/dev.yaml`：

```yaml
model:
  download_source: auto
```

### ModelDownloader 改动

```python
class ModelDownloader:
    def __init__(self, cache_dir: Path, max_retries: int = 3,
                 hf_token: str | None = None,
                 endpoint: str | None = None,
                 download_source: str = "auto"):
        self._cache_dir = cache_dir
        self._strategy = self._build_strategy(download_source)

    def _build_strategy(self, source: str) -> DownloadStrategy:
        hf = HfStrategy(self._endpoint, self._hf_token, self._max_retries)
        if source == "huggingface":
            return hf
        ms = MsStrategy()
        if source == "modelscope":
            return ms
        if source == "auto":
            return AutoStrategy(ms, hf)
        raise ValueError(f"不支持的 download_source: {source}")

    def download(self, model_id: str, force: bool = False) -> Path:
        model_id = _validate_model_id(model_id)
        if not force and self.is_downloaded(model_id):
            return self.model_dir(model_id)
        return self._strategy.download(model_id, force, self._cache_dir)
```

### ModelManager 改动

`_ensure_init()` 传递 `download_source`：

```python
self._downloader = ModelDownloader(
    cache_dir=cache_dir,
    max_retries=cfg.max_retries,
    hf_token=token,
    endpoint=cfg.hf_endpoint,
    download_source=cfg.download_source,  # 新增
)
```

### 消除的代码

- 独立函数 `download_from_modelscope()` → 删除，逻辑移入 `MsStrategy`
- 模块级 `_MODELSCOPE_AVAILABLE` → 删除，检查移入 `MsStrategy.__init__`
- `if __name__ == "__main__"` 中 modelscope 分支 → 改为统一入口

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/model/downloader.py` | 修改 | 新增策略类，重构 `ModelDownloader`，删除独立函数 |
| `src/model/__init__.py` | 修改 | 调整导出 |
| `src/config/settings.py` | 修改 | `ModelConfig` 新增 `download_source` 字段 |
| `config/dev.yaml` | 修改 | 新增 `model.download_source: auto` |
| `src/model/manager.py` | 修改 | 传递 `download_source` 参数 |
| `src/model/README.md` | 修改 | 更新文档 |

## 约束

- 公共接口不变：`ModelDownloader.download()` 签名和返回值不变
- `ModelManager` 对外 API 完全不变
- `modelscope` 库为可选依赖，未安装时 MsStrategy 抛 `RuntimeError`
- `download_source` 不合法值时 `_build_strategy` 抛 `ValueError`

## 风险

- **低风险**：纯重构，不改变公共 API，下游零影响
- `AutoStrategy` 在魔搭失败时日志记录后回退，可能略微增加首次下载延迟（网络超时时间）
