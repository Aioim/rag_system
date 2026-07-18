"""LLMRouter — 纯规则模型路由 + 按意图加载 Prompt 模板

路由规则（设计文档 5.6，本期纯规则实现）：
- lookup / procedure → lightweight（精确查找/步骤类问题用轻量模型）
- concept / compare  → default（概念/对比类问题用大模型）
"""

from dataclasses import dataclass

import yaml

from config import settings
from config.path import PROJECT_ROOT
from models.enums import Intent

# intent → 模型档位
_TIER_BY_INTENT = {
    Intent.LOOKUP: "lightweight",
    Intent.PROCEDURE: "lightweight",
    Intent.CONCEPT: "default",
    Intent.COMPARE: "default",
}

_FALLBACK_INTENT = Intent.CONCEPT


@dataclass
class RouteResult:
    """路由决策结果"""
    model_tier: str        # "default" | "lightweight"
    model_name: str
    temperature: float
    system_prompt: str
    user_template: str


class LLMRouter:
    """route(intent) → RouteResult；模板按 intent 懒加载并缓存"""

    def __init__(self):
        self._prompts_dir = PROJECT_ROOT / "config" / "prompts"
        self._templates: dict[Intent, dict] = {}

    def _load_template(self, intent: Intent) -> dict:
        cached = self._templates.get(intent)
        if cached is not None:
            return cached

        path = self._prompts_dir / f"{intent.value}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt 模板不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            template = yaml.safe_load(f) or {}
        self._templates[intent] = template
        return template

    def route(self, intent: Intent | None) -> RouteResult:
        """按 intent 决定模型档位/温度/Prompt 模板；intent 缺失时降级 CONCEPT"""
        if intent not in _TIER_BY_INTENT:
            intent = _FALLBACK_INTENT

        tier = _TIER_BY_INTENT[intent]
        llm_cfg = settings.llm
        template = self._load_template(intent)
        return RouteResult(
            model_tier=tier,
            model_name=llm_cfg.lightweight if tier == "lightweight" else llm_cfg.default,
            temperature=llm_cfg.temperatures.get(intent.value, 0.0),
            system_prompt=template.get("system", ""),
            user_template=template.get("user_template", ""),
        )
