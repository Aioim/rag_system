"""
demo_config.py — 配置管理模块演示

演示内容：
  1. 基础配置访问（属性 / 点号路径 / get 方法）
  2. 配置段结构与默认值
  3. 环境变量覆盖机制
  4. CLI 参数覆盖（apply_overrides）
  5. 配置热重载（reload）
  6. 项目路径工具（PROJECT_ROOT）

运行方式：
  cd rag0709
  python examples/01_config/demo_config.py

可选：通过环境变量验证覆盖机制
  # Windows PowerShell
  $env:RETRIEVAL__TOP_K="20"
  python examples/01_config/demo_config.py
  Remove-Item Env:\RETRIEVAL__TOP_K

  # Linux/macOS
  RETRIEVAL__TOP_K=20 python examples/01_config/demo_config.py
"""

import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    from config import settings, PROJECT_ROOT

    # ── 1. 项目根路径 ──────────────────────────────────────────
    banner("1. 项目根路径 (PROJECT_ROOT)")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  类型: {type(PROJECT_ROOT).__name__}")
    print(f"  是否存在: {PROJECT_ROOT.exists()}")

    # ── 2. 基础配置访问 ────────────────────────────────────────
    banner("2. 基础配置访问")

    # 属性访问
    print(f"  settings.retrieval.top_k          = {settings.retrieval.top_k}")
    print(f"  settings.retrieval.rrf_k          = {settings.retrieval.rrf_k}")
    print(f"  settings.retrieval.mmr_lambda     = {settings.retrieval.mmr_lambda}")

    # 点号路径访问
    print(f"  settings.get('retrieval.top_k')   = {settings.get('retrieval.top_k')}")
    print(f"  settings.get('llm.default')       = {settings.get('llm.default')}")
    print(f"  settings.get('chunking.strategy') = {settings.get('chunking.strategy')}")

    # 不存在的键
    result = settings.get("nonexistent.key", "默认值")
    print(f"  settings.get('nonexistent.key', '默认值') = {result!r}")

    # ── 3. 各配置段概览 ────────────────────────────────────────
    banner("3. 配置段概览")

    config_sections = [
        ("项目信息", lambda: f"name={settings.project.name}, version={settings.project.version}"),
        ("API 服务", lambda: f"host={settings.api.host}:{settings.api.port}, workers={settings.api.workers}"),
        ("检索配置", lambda: f"top_k={settings.retrieval.top_k}, rrf_k={settings.retrieval.rrf_k}, mmr={settings.retrieval.mmr_lambda}"),
        ("文档分块", lambda: f"size={settings.chunking.chunk_size}, overlap={settings.chunking.overlap}, strategy={settings.chunking.strategy}"),
        ("Embedding", lambda: f"model={settings.embedding.model}, dim={settings.embedding.dimension}"),
        ("LLM 路由", lambda: f"default={settings.llm.default}, lightweight={settings.llm.lightweight}"),
        ("生成层", lambda: f"dedup={settings.generation.dedup_threshold}, max_chars={settings.generation.max_context_chars}"),
        ("联网搜索", lambda: f"enabled={settings.web_search.enabled}, provider={settings.web_search.provider}"),
        ("兜底策略", lambda: f"max_rounds={settings.fallback.max_retrieval_rounds}"),
        ("会话管理", lambda: f"ttl={settings.session.ttl_hours}h, max_rounds={settings.session.max_history_rounds}"),
        ("日志", lambda: f"level={settings.log.log_level}, dir={settings.log.log_dir}"),
        ("模型下载", lambda: f"cache_dir={settings.model.cache_dir}, source={settings.model.download_source}"),
        ("FAISS", lambda: f"type={settings.faiss.index_type}, dim={settings.faiss.dimension}"),
        ("微调", lambda: f"epochs={settings.finetune.training.epochs}, lora_r={settings.finetune.lora.r}"),
        ("Agent", lambda: f"max_iter={settings.agent.max_iterations}, top_k={settings.agent.search_top_k}"),
    ]

    for label, getter in config_sections:
        try:
            print(f"  [{label}] {getter()}")
        except Exception as e:
            print(f"  [{label}] ⚠️ 获取失败: {e}")

    # ── 4. CLI 参数覆盖 ────────────────────────────────────────
    banner("4. CLI 参数覆盖 (apply_overrides)")

    original_top_k = settings.retrieval.top_k
    print(f"  覆盖前: top_k = {original_top_k}")
    settings.apply_overrides("retrieval.top_k=50;retrieval.rrf_k=80;retrieval.mmr_lambda=0.9")
    print(f"  覆盖后: top_k = {settings.retrieval.top_k}")
    print(f"          rrf_k = {settings.retrieval.rrf_k}")
    print(f"          mmr_lambda = {settings.retrieval.mmr_lambda}")

    # ── 5. 配置热重载 ──────────────────────────────────────────
    banner("5. 配置热重载 (reload)")

    old_top_k = settings.retrieval.top_k
    print(f"  重载前: top_k = {old_top_k}")
    settings.reload()
    new_top_k = settings.retrieval.top_k
    print(f"  重载后: top_k = {new_top_k}")
    print(f"  说明: reload() 会清空 CLI 覆盖，恢复为 YAML 原始值")

    # ── 6. 环境变量注入验证 ────────────────────────────────────
    banner("6. 环境变量注入（白名单）")

    print(f"  ENV 环境变量: {os.environ.get('ENV', '(未设置)')}")
    print(f"  DEBUG 环境变量: {os.environ.get('DEBUG', '(未设置)')}")
    print(f"  LLM__DEFAULT 环境变量: {os.environ.get('LLM__DEFAULT', '(未设置)')}")
    print()
    print("  环境变量命名规则:")
    print("    RETRIEVAL__TOP_K=10        → retrieval.top_k = 10")
    print("    LLM__DEFAULT=deepseek-pro → llm.default = 'deepseek-pro'")
    print("    RAG__CUSTOM__KEY=val      → custom.key = 'val' (RAG__ 逃生前缀)")

    # ── 7. 配置模型验证 ────────────────────────────────────────
    banner("7. 配置模型验证示例")

    print("  所有配置段都经过 Pydantic v2 校验：")
    print(f"  ✓ retrieval.top_k >= 1:        {settings.retrieval.top_k}")
    print(f"  ✓ retrieval.mmr_lambda ∈ [0,1]: {settings.retrieval.mmr_lambda}")
    print(f"  ✓ chunking.chunk_size > 0:     {settings.chunking.chunk_size}")

    # ── 8. 敏感信息保护 ────────────────────────────────────────
    banner("8. 敏感信息保护验证")

    print(f"  llm.api_key_env = {settings.llm.api_key_env}")
    print(f"  model.hf_token_env = {settings.model.hf_token_env}")
    print(f"  LLM_API_KEY 在 .env 中: {'已设置' if os.environ.get('LLM_API_KEY') else '(未设置)'}")
    print()
    print("  ⚠️  YAML 配置文件中禁止写入任何密钥/密码/Token！")
    print("  敏感信息仅通过 .env 文件配置，支持 Fernet 加密: ENC[...]")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 配置模块演示完成")
    print()
    print("  扩展阅读:")
    print("    - 配置优先级: CLI覆盖 > 环境变量 > {env}.yaml > 代码默认值")
    print("    - 环境切换: ENV=prod python examples/01_config/demo_config.py")
    print("    - 新增配置: 在 config/dev.yaml 中添加新段，对应 Pydantic 模型")


if __name__ == "__main__":
    main()
