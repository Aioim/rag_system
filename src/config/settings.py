"""
RAG 企业级知识库问答系统 — 配置管理模块
支持：YAML 基础配置 → 环境变量覆盖（双下划线表示嵌套）→ 命令行覆盖
"""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List, ClassVar, Set

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, SecretStr
from config.path import PROJECT_ROOT
from config.yaml_loader import YamlLoader, deep_merge


# ============================================================================
# 配置模型定义
# ============================================================================

class _BaseConfig(BaseModel):
    """配置模型基类：允许额外字段"""
    model_config = ConfigDict(extra="allow")


class ProjectConfig(_BaseConfig):
    """项目基础配置"""
    name: str = "rag-enterprise-qa"
    root: Path = PROJECT_ROOT
    version: str = "1.0.0"


class APIConfig(_BaseConfig):
    """API 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    workers: int = 1


class RetrievalConfig(_BaseConfig):
    """检索配置"""
    top_k: int = Field(default=5, ge=1)
    expansion_window: int = 1
    rrf_k: int = Field(default=60, gt=0)
    max_rerank_candidates: int = 30
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    relevance_threshold_sufficient: float = 0.5
    relevance_threshold_need_more: float = 0.3
    similarity_dedup_threshold: float = 0.85
    max_context_tokens: int = 6000

    @model_validator(mode="after")
    def check_threshold_order(self) -> "RetrievalConfig":
        """阈值顺序反转会使 Self-RAG 自评逻辑静默出错，配置加载时尽早暴露"""
        if self.relevance_threshold_sufficient < self.relevance_threshold_need_more:
            raise ValueError(
                f"relevance_threshold_sufficient ({self.relevance_threshold_sufficient}) "
                f"必须 >= relevance_threshold_need_more ({self.relevance_threshold_need_more})"
            )
        return self


class ChunkingConfig(_BaseConfig):
    """文档分块配置"""
    chunk_size: int = 512
    overlap: int = 64
    strategy: str = "semantic"
    semantic_threshold_percentile: float = 0.9
    semantic_buffer_size: int = 1


class SessionConfig(_BaseConfig):
    """会话管理配置"""
    ttl_hours: int = 2
    max_history_rounds: int = 10
    max_context_tokens: int = 4000
    db_path: Path = PROJECT_ROOT / "data" / "sessions.db"
    topic_switch_threshold: float = 0.5
    cleanup_interval_seconds: int = 600

    def initialize(self) -> None:
        """创建 db_path 父目录"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 无法创建数据目录: {e}", file=sys.stderr)


class EmbeddingConfig(_BaseConfig):
    """Embedding 模型配置"""
    model: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    dimension: int = 1024


class LLMConfig(_BaseConfig):
    """LLM 路由与 API 配置

    安全约束：api_key 只能通过环境变量或 .env 文件设置，不允许在 YAML 中配置。
    使用 encrypted .env 时，值格式为 ENC[base64_ciphertext]。
    """
    default: str = "claude-sonnet-5"
    lightweight: str = "claude-haiku-4-5"
    local: Optional[str] = None
    api_key: SecretStr = Field(default=SecretStr(""), exclude=True)
    api_key_env: str = "LLM_API_KEY"
    api_base_url: Optional[str] = None
    temperatures: Dict[str, float] = Field(default_factory=lambda: {
        "concept": 0.3, "procedure": 0.0, "compare": 0.2, "lookup": 0.0,
    })

    @field_validator("api_key")
    @classmethod
    def reject_yaml_secret(cls, v: SecretStr) -> SecretStr:
        """禁止在 YAML 中配置 api_key ─ 密钥只能通过环境变量 / .env 设置"""
        if v.get_secret_value():
            raise ValueError(
                "api_key 不允许在 YAML 中配置！"
                "请通过环境变量 LLM_API_KEY 设置，或写入 .env 文件"
                "（支持 ENC[...] Fernet 加密格式，使用 python -m security.env_encrypt 生成）"
            )
        return v

    @model_validator(mode="after")
    def resolve_api_key(self) -> "LLMConfig":
        """从环境变量回填 api_key（字段校验器保证 YAML 侧一定为空）"""
        if not self.api_key.get_secret_value():
            env_val = os.getenv(self.api_key_env, "")
            if env_val:
                object.__setattr__(self, "api_key", SecretStr(env_val))
        return self


class GenerationConfig(_BaseConfig):
    """生成层配置"""
    dedup_threshold: float = Field(default=0.85, ge=0.0, le=1.0)   # 上下文去重余弦阈值
    max_context_chars: int = Field(default=9000, ge=1)             # 上下文字符预算（近似 6000 tokens）
    max_query_chars: int = Field(default=2000, ge=1)               # 用户 query 截断上限（防 DoS）
    fact_check_enabled: bool = True                                # 事实核查开关


class WebSearchConfig(_BaseConfig):
    """联网搜索兜底配置"""
    enabled: bool = True
    provider: str = "duckduckgo"
    timeout_seconds: int = 10


class MilvusConfig(_BaseConfig):
    """Milvus 向量数据库配置"""
    host: str = "localhost"
    port: int = 19530
    collections: Dict[str, str] = Field(default_factory=lambda: {"default": "rag_default"})
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 1024


class FallbackConfig(_BaseConfig):
    """兜底策略配置"""
    max_retrieval_rounds: int = 2
    no_answer_message: str = (
        "抱歉，当前知识库中未找到相关信息，建议补充相关文档或联系管理员。"
    )


class AliasConfig(_BaseConfig):
    """术语别名映射配置"""
    file_path: Path = PROJECT_ROOT / "config" / "aliases.yaml"
    auto_reload: bool = True


class ModelConfig(_BaseConfig):
    """模型下载管理配置"""
    cache_dir: Path = PROJECT_ROOT / "models"
    default_models: Dict[str, str] = Field(default_factory=lambda: {
        "embedding": "BAAI/bge-large-zh-v1.5",
        "rerank": "BAAI/bge-reranker-v2-m3",
        "llm": "Qwen/Qwen2.5-1.5B-Instruct",
    })
    hf_token_env: str = "HUGGINGFACE_TOKEN"
    hf_endpoint: Optional[str] = "https://hf-mirror.com"
    max_retries: int = 3


class LogConfig(_BaseConfig):
    """日志配置 — 与 logger 模块兼容"""
    log_dir: Path = PROJECT_ROOT / "logs"
    log_level: str = "INFO"
    log_file: str = "rag_service.log"
    backup_count: int = 7
    max_bytes: int = 10 * 1024 * 1024          # 10MB
    enable_colors: bool = False
    quiet: bool = False
    replace_main_with_filename: bool = True

    SENSITIVE_KEYS: ClassVar[Set[str]] = {
        "password", "pwd", "pass", "secret", "token", "api_key", "apikey",
        "authorization", "cookie", "x-api-key", "access_token", "refresh_token",
        "new_password", "old_password", "confirm_password", "credit_card",
        "ssn", "social_security", "passport", "cvv", "pin", "private_key"
    }

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"无效日志级别: {v}, 必须是 {valid}")
        return v

    def initialize(self) -> None:
        """初始化日志目录（由日志模块调用）"""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ 无法创建日志目录: {e}", file=sys.stderr)

    model_config = ConfigDict(protected_namespaces=())


class FaissConfig(_BaseConfig):
    """FAISS 向量数据库配置（第一期）"""
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 100
    nprobe: int = 10
    dimension: int = 1024
    index_dir: Path = PROJECT_ROOT / "data" / "faiss_indexes"

    def initialize(self) -> None:
        """创建索引持久化目录"""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 无法创建 FAISS 索引目录: {e}", file=sys.stderr)


class FinetuneTrainingConfig(_BaseConfig):
    """微调训练超参数"""
    epochs: int = 3
    learning_rate: float = 2.0e-4
    batch_size: int = 8
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50


class FinetuneLoraConfig(_BaseConfig):
    """LoRA 适配器配置"""
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None


class FinetuneDistillationConfig(_BaseConfig):
    """知识蒸馏配置"""
    temperature: float = 2.0
    alpha: float = 0.5


class FinetuneConfig(_BaseConfig):
    """模型微调 & 蒸馏配置"""
    output_dir: Path = PROJECT_ROOT / "models" / "finetuned"
    device: str = "auto"
    data_dir: Path = PROJECT_ROOT / "data" / "finetune"
    training: FinetuneTrainingConfig = Field(default_factory=FinetuneTrainingConfig)
    lora: FinetuneLoraConfig = Field(default_factory=FinetuneLoraConfig)
    distillation: FinetuneDistillationConfig = Field(default_factory=FinetuneDistillationConfig)


class RAGAppConfig(BaseModel):
    """RAG 应用主配置 — 支持环境变量覆盖（双下划线表示嵌套）"""
    # 核心
    env: str = "dev"
    debug: bool = False
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    # 各子系统配置
    api: APIConfig = Field(default_factory=APIConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    aliases: AliasConfig = Field(default_factory=AliasConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    faiss: FaissConfig = Field(default_factory=FaissConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        return v.lower()

    # 系统环境变量前缀（不会被注入配置）。
    # 注意：此处采用黑名单方式过滤已知系统变量，覆盖 Windows / Linux / macOS / CI 常见变量。
    # 若新增配置环境变量与下列前缀冲突，请在自定环境变量前加 RAG__ 前缀以避开过滤。
    _SYSTEM_ENV_PREFIXES: ClassVar[Set[str]] = {
        # POSIX / shell
        "path", "home", "user", "shell", "term", "lang", "lc_", "pwd", "oldpwd",
        "editor", "display", "xdg", "dbus", "desktop", "wayland", "ssh_", "gpg_",
        "colour", "colorterm", "vte", "tmux", "iter", "logname", "mail", "hostname",
        "ps1", "ps2", "ps4", "ifs", "hist", "shlvl", "tty",
        # Windows
        "temp", "tmp", "windir", "os", "systemroot", "systemdrive", "homedrive",
        "computername", "username", "userprofile", "allusersprofile", "userdomain",
        "programfiles", "commonprogramfiles", "programdata", "appdata", "localappdata",
        "onedrive", "driverdata", "number_of_processors", "processor",
        "sessionname", "logonserver", "public", "psmodulepath", "pathext",
        "comspec", "commonprogramw6432", "programw6432",
        # 通用 / CI / 容器
        "ci", "build_", "jenkins", "github_", "gitlab", "travis", "circle",
        "docker", "kubernetes", "kube", "nomad", "container",
        "java_home", "conda", "virtual_env", "python", "pip", "pyenv",
        "npm_", "node_", "yarn_", "cargo", "rustup", "gopath", "goroot",
        "aws_", "azure_", "gcloud", "google_application",
        "no_proxy", "http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
        "ssl_", "tls_", "require_https",
    }

    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """
        从环境变量构建配置字典（过滤系统变量）。
        支持嵌套：双下划线 __ 表示嵌套层级。
        例如 RETRIEVAL__TOP_K=10 → retrieval.top_k = 10

        环境变量键名不区分大小写。所有键统一转为小写处理。
        若自定义环境变量被系统黑名单误过滤，可加 RAG__ 前缀：
        例如 RAG__MY_KEY=val → my_key = val
        """
        env_data: Dict[str, Any] = {}
        for key, value in os.environ.items():
            clean_key = key.lower()
            # 允许 rag__ 前缀以绕过系统变量黑名单过滤
            has_rag_prefix = clean_key.startswith("rag__")
            if has_rag_prefix:
                clean_key = clean_key[5:]  # 去掉 "rag__" 前缀
            # 跳过系统环境变量（但 rag__ 前缀的变量不受此限制）
            if not has_rag_prefix and any(clean_key.startswith(p) for p in cls._SYSTEM_ENV_PREFIXES):
                continue
            if "__" in clean_key:
                parts = clean_key.split("__")
                current = env_data
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = cls._parse_env_value(value)
            else:
                env_data[clean_key] = cls._parse_env_value(value)
        return env_data

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """将环境变量字符串转换为 bool 或 int；其余保留字符串交由 Pydantic 处理"""
        low = value.lower()
        if low in ("true", "false"):
            return low == "true"
        # 仅对纯整数格式做转换（不含小数点，避免 "1.0" 被误判为 float）
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return value

    model_config = ConfigDict(protected_namespaces=(), extra="allow")


# ============================================================================
# 配置管理器（单例）
# ============================================================================

class ConfigManager:
    """统一配置管理器 — 聚合 YAML、环境变量、命令行覆盖"""

    _instance: ClassVar[Optional["ConfigManager"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._config: Optional[RAGAppConfig] = None
                    cls._instance._yaml_loader = YamlLoader()
                    cls._instance._overrides: Dict[str, Any] = {}
                    cls._instance._dict_cache: Optional[Dict[str, Any]] = None
                    cls._instance._initialized = False
        return cls._instance

    def _load_full_config(self) -> RAGAppConfig:
        """加载并合并所有配置源"""
        # 1. 加载 YAML 基础配置（按环境）
        env_name = self._overrides.get("env") or os.getenv("ENV", "dev")
        yaml_config = self._yaml_loader.load_environment(env=env_name)

        # 2. 从环境变量获取覆盖字典
        env_overrides = RAGAppConfig.from_env()

        # 3. 深度合并：YAML 被环境变量覆盖
        merged_dict = deep_merge(yaml_config, env_overrides)

        # 4. 应用命令行覆盖（优先级最高）
        final_dict = deep_merge(merged_dict, self._overrides)

        # 5. 验证并返回最终配置
        return RAGAppConfig(**final_dict)

    def initialize(self) -> None:
        """显式初始化（通常由属性访问自动触发）"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    try:
                        self._config = self._load_full_config()
                        self._config.log.initialize()
                        self._config.session.initialize()
                        self._config.faiss.initialize()
                        # 从配置路径加载别名（若与默认路径不同）
                        from config.aliases import alias_manager as _am
                        _am.load(self._config.aliases.file_path)
                        self._initialized = True
                    except Exception as e:
                        raise RuntimeError(f"配置加载失败: {e}") from e

    def __getattr__(self, name: str) -> Any:
        if self._config is None:
            self.initialize()
        try:
            return getattr(self._config, name)
        except AttributeError:
            available = [attr for attr in dir(self._config) if not attr.startswith("_")]
            raise AttributeError(
                f"配置中不存在属性 '{name}'. 可用属性: {', '.join(available[:20])}"
            )

    def __repr__(self) -> str:
        if self._config is None:
            return "<ConfigManager (uninitialized)>"
        return f"<ConfigManager env={self._config.env} debug={self._config.debug}>"

    def get(self, path: str, default: Any = None) -> Any:
        """通过点号路径获取嵌套配置，如 settings.get('retrieval.top_k')"""
        if self._config is None:
            self.initialize()
        if self._dict_cache is None:
            self._dict_cache = self._config.model_dump()
        current: Any = self._dict_cache
        for key in path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def apply_overrides(self, overrides_str: str) -> None:
        """解析命令行覆盖字符串，以分号分隔: 'key=value;key2.subkey=value2'

        若已初始化则自动重载使覆盖生效。
        """
        if not overrides_str:
            return
        new_overrides: Dict[str, Any] = {}
        pairs = overrides_str.split(";")
        for pair in pairs:
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            keys = [k.strip() for k in key.strip().split(".")]
            current = new_overrides
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = self._parse_override_value(value.strip())
        # 累加合并，不覆盖之前的 overrides
        self._overrides = deep_merge(self._overrides, new_overrides)
        # 已初始化则自动重载
        if self._initialized:
            self.reload()

    @classmethod
    def _parse_override_value(cls, value: str) -> Any:
        """解析命令行覆盖的值（支持布尔、数字、列表、简单字典）"""
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            return [cls._parse_override_value(it.strip()) for it in items if it.strip()]
        if value.startswith("{") and value.endswith("}"):
            result = {}
            content = value[1:-1].strip()
            if content:
                for item in content.split(","):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        result[k.strip().strip('"\'')] = cls._parse_override_value(v.strip())
            return result
        # 标量委托给 _parse_env_value
        return RAGAppConfig._parse_env_value(value)

    def load_from_file(self, config_path: Path) -> None:
        """直接加载指定的配置文件，并叠加环境变量覆盖"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 叠加环境变量覆盖和命令行覆盖
        merged = deep_merge(data, RAGAppConfig.from_env())
        merged = deep_merge(merged, self._overrides)
        self._config = RAGAppConfig(**merged)
        self._dict_cache = None
        self._config.log.initialize()
        self._config.session.initialize()
        self._config.faiss.initialize()
        self._initialized = True

    def to_yaml(self) -> str:
        """导出当前配置为 YAML（隐藏敏感字段）"""
        if self._config is None:
            self.initialize()
        data = self._config.model_dump(exclude_none=False)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def reload(self) -> None:
        """热重载配置（清空缓存并重新加载）"""
        self._yaml_loader.clear_cache()
        with self._lock:
            self._config = None
            self._dict_cache = None
            self._initialized = False
        self.initialize()
        # 别名文件可能已变更，强制重新加载
        from config.aliases import alias_manager as _am
        _am.reload()


# ============================================================================
# 全局单例导出
# ============================================================================

settings = ConfigManager()

__all__ = [
    "settings",
    "ConfigManager",
    "RAGAppConfig",
    "ProjectConfig",
    "APIConfig",
    "RetrievalConfig",
    "ChunkingConfig",
    "SessionConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "GenerationConfig",
    "WebSearchConfig",
    "MilvusConfig",
    "FallbackConfig",
    "AliasConfig",
    "ModelConfig",
    "LogConfig",
    "FaissConfig",
    "FinetuneConfig",
    "FinetuneTrainingConfig",
    "FinetuneLoraConfig",
    "FinetuneDistillationConfig",
]

if __name__ == "__main__":
    print("=== RAG 配置测试 ===")
    print(f"项目: {settings.project.name} v{settings.project.version}")
    print(f"环境: {settings.env}")
    print(f"API: {settings.api.host}:{settings.api.port}")
    print(f"检索 Top-K: {settings.retrieval.top_k}")
    print(f"分块大小: {settings.chunking.chunk_size}")
    print(f"Embedding 模型: {settings.embedding.model}")
    print(f"LLM 默认: {settings.llm.default}")
    print(f"Milvus: {settings.milvus.host}:{settings.milvus.port}")
    print(f"日志级别: {settings.log.log_level}")
