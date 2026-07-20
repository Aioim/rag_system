"""
RAG 企业级知识库问答系统 — 配置管理模块
支持：YAML 基础配置 → 环境变量覆盖（双下划线表示嵌套）→ 命令行覆盖
"""

import os
import threading
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from config.path import PROJECT_ROOT
from config.yaml_loader import YamlLoader, deep_merge

# ============================================================================
# 配置模型定义
# ============================================================================

class _BaseConfig(BaseModel):
    """配置模型基类：允许额外字段；相对路径统一归一化到 PROJECT_ROOT"""
    model_config = ConfigDict(extra="allow")

    @field_validator("*", mode="after")
    @classmethod
    def _resolve_relative_path(cls, v: Any) -> Any:
        """YAML 中的相对路径（如 db_path: data/sessions.db）锚定到
        PROJECT_ROOT，消除对进程启动目录（CWD）的依赖"""
        if isinstance(v, Path) and not v.is_absolute():
            return PROJECT_ROOT / v
        return v


class ProjectConfig(_BaseConfig):
    """项目基础配置"""
    name: str = "rag-enterprise-qa"
    root: Path = PROJECT_ROOT
    version: str = "1.0.0"


class APIConfig(_BaseConfig):
    """API 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    # 默认 "*" 仅适用于开发环境；生产环境由 RAGAppConfig 的 model_validator 阻止
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


class MinerUConfig(BaseModel):
    """MinerU 解析器配置"""
    device: str = "cpu"  # cpu | cuda | mps
    models_dir: str = "local_models/mineru"


class IngestionConfig(_BaseConfig):
    """文档处理（离线 Pipeline）配置"""
    parsed_doc_dir: Path = PROJECT_ROOT / "data" / "parsed_docs"
    mineru: MinerUConfig = Field(default_factory=MinerUConfig)
    parsers: dict[str, str] = Field(default_factory=lambda: {
        "pdf": "docling",
        "docx": "docling",
        "doc": "docling",
        "pptx": "docling",
        "ppt": "docling",
        "html": "docling",
        "md": "direct",
        "markdown": "direct",
        "txt": "direct",
    })

    def initialize(self) -> None:
        """创建解析文档输出目录"""
        try:
            self.parsed_doc_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning(
                "无法创建解析文档目录 %s: %s", self.parsed_doc_dir, e
            )


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
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning(
                "无法创建数据目录 %s: %s", self.db_path.parent, e
            )


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
    local: str | None = None
    api_key: SecretStr = Field(default=SecretStr(""), exclude=True)
    api_key_env: str = "LLM_API_KEY"
    api_base_url: str | None = None
    temperatures: dict[str, float] = Field(default_factory=lambda: {
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
    collections: dict[str, str] = Field(default_factory=lambda: {"default": "rag_default"})
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
    cache_dir: Path = PROJECT_ROOT / "local_models"
    default_models: dict[str, str] = Field(default_factory=lambda: {
        "embedding": "BAAI/bge-large-zh-v1.5",
        "rerank": "BAAI/bge-reranker-v2-m3",
        "llm": "Qwen/Qwen2.5-1.5B-Instruct",
    })
    hf_token_env: str = "HUGGINGFACE_TOKEN"
    hf_endpoint: str | None = "https://hf-mirror.com"
    max_retries: int = 3
    download_source: str = "auto"  # "huggingface" | "modelscope" | "auto"


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

    SENSITIVE_KEYS: ClassVar[set[str]] = {
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
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning(
                "无法创建日志目录 %s: %s", self.log_dir, e
            )

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
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning(
                "无法创建 FAISS 索引目录 %s: %s", self.index_dir, e
            )


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
    target_modules: list[str] | None = None


class FinetuneDistillationConfig(_BaseConfig):
    """知识蒸馏配置"""
    temperature: float = 2.0
    alpha: float = 0.5


class FinetuneConfig(_BaseConfig):
    """模型微调 & 蒸馏配置"""
    output_dir: Path = PROJECT_ROOT / "local_models" / "finetuned"
    device: str = "auto"
    data_dir: Path = PROJECT_ROOT / "data" / "finetune"
    training: FinetuneTrainingConfig = Field(default_factory=FinetuneTrainingConfig)
    lora: FinetuneLoraConfig = Field(default_factory=FinetuneLoraConfig)
    distillation: FinetuneDistillationConfig = Field(default_factory=FinetuneDistillationConfig)


class AgentConfig(_BaseConfig):
    """ReAct Agent 配置"""
    max_iterations: int = Field(default=5, ge=1, le=20)
    search_top_k: int = Field(default=3, ge=1, le=10)
    max_observation_chars: int = Field(default=3000, ge=100, le=10000)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_consecutive_duplicates: int = Field(default=2, ge=1, le=5)


class RAGAppConfig(BaseModel):
    """RAG 应用主配置 — 支持环境变量覆盖（双下划线表示嵌套）"""
    # 核心
    env: str = "dev"
    debug: bool = False
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    # 各子系统配置
    api: APIConfig = Field(default_factory=APIConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
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
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        return v.lower()

    @model_validator(mode="after")
    def check_cors_in_prod(self) -> "RAGAppConfig":
        """生产环境禁止 CORS 通配符 '*'，须显式配置业务域名"""
        if self.env == "prod" and "*" in self.api.cors_origins:
            raise ValueError(
                "api.cors_origins 不允许在生产环境使用通配符 '*'，"
                "请配置具体业务域名，例如 API__CORS_ORIGINS=https://your-app.example.com"
            )
        return self

    # 无嵌套（不含 __）时允许注入的顶层标量键
    _TOP_LEVEL_ENV_KEYS: ClassVar[set[str]] = {"env", "debug"}

    @classmethod
    def from_env(cls) -> dict[str, Any]:
        """
        从环境变量构建配置字典（白名单过滤）。
        支持嵌套：双下划线 __ 表示嵌套层级。
        例如 RETRIEVAL__TOP_K=10 → retrieval.top_k = 10

        过滤规则（白名单，避免系统/无关变量误入配置）：
        1. RAG__ 前缀：无条件注入（去掉前缀），如 RAG__MY_KEY=val → my_key
        2. 含 __ 的嵌套变量：根段必须是已声明的配置段（retrieval/llm/...）
        3. 无 __ 的变量：仅允许 ENV / DEBUG 顶层标量

        环境变量键名不区分大小写。所有键统一转为小写处理。
        """
        allowed_roots = set(cls.model_fields.keys())
        env_data: dict[str, Any] = {}
        for key, value in os.environ.items():
            clean_key = key.lower()
            # rag__ 前缀无条件放行（去掉前缀后注入）
            has_rag_prefix = clean_key.startswith("rag__")
            if has_rag_prefix:
                clean_key = clean_key[5:]  # 去掉 "rag__" 前缀
            if "__" in clean_key:
                parts = clean_key.split("__")
                if not has_rag_prefix and parts[0] not in allowed_roots:
                    continue
                current = env_data
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = cls._parse_env_value(value)
            else:
                if not has_rag_prefix and clean_key not in cls._TOP_LEVEL_ENV_KEYS:
                    continue
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

    # 实例属性类型声明（在 __new__ 中初始化，此处为 mypy 类型追踪）
    _config: RAGAppConfig | None = None
    _yaml_loader: Any = None  # YamlLoader，延迟引用避免循环
    _overrides: dict[str, Any] = {}
    _dict_cache: dict[str, Any] | None = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._config = None
                    cls._instance._yaml_loader = YamlLoader()
                    cls._instance._overrides = {}
                    cls._instance._dict_cache = None
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
                        self._config.ingestion.initialize()
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
            ) from None

    def __repr__(self) -> str:
        if self._config is None:
            return "<ConfigManager (uninitialized)>"
        return f"<ConfigManager env={self._config.env} debug={self._config.debug}>"

    def get(self, path: str, default: Any = None) -> Any:
        """通过点号路径获取嵌套配置，如 settings.get('retrieval.top_k')"""
        if self._config is None:
            self.initialize()
        # _dict_cache 的惰性构建/读取与 reload() 竞争，需在锁内取快照
        with self._lock:
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
        new_overrides: dict[str, Any] = {}
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
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 叠加环境变量覆盖和命令行覆盖
        merged = deep_merge(data, RAGAppConfig.from_env())
        merged = deep_merge(merged, self._overrides)
        self._config = RAGAppConfig(**merged)
        self._dict_cache = None
        self._config.ingestion.initialize()
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
    "AgentConfig",
    "APIConfig",
    "AliasConfig",
    "ChunkingConfig",
    "ConfigManager",
    "EmbeddingConfig",
    "FaissConfig",
    "FallbackConfig",
    "FinetuneConfig",
    "FinetuneDistillationConfig",
    "FinetuneLoraConfig",
    "FinetuneTrainingConfig",
    "GenerationConfig",
    "IngestionConfig",
    "LLMConfig",
    "LogConfig",
    "MilvusConfig",
    "ModelConfig",
    "ProjectConfig",
    "RAGAppConfig",
    "RetrievalConfig",
    "SessionConfig",
    "WebSearchConfig",
    "settings",
]
