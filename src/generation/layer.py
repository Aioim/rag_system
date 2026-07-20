"""GenerationLayer — 生成层主编排器

流程（设计文档 5.5–5.7）：
1. INSUFFICIENT → 短路（不调 LLM，写 fallback_level=FallbackLevel.NO_ANSWER，上层 core/fallback 处理兜底）
2. NEED_MORE → 标记 fallback_level=FallbackLevel.PARTIAL，正常生成
3. SUFFICIENT → 完整流程：组装 → 路由 → 生成 → 核查 → 引用
"""
import time

from config import settings
from generation.citation_builder import CitationBuilder
from generation.fact_checker import FactChecker, FactCheckResult
from generation.llm_router import LLMRouter
from generation.prompt_assembler import PromptAssembler
from logger import logger
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval
from models.llm import LLMProtocol


class GenerationLayer:
    """构造时注入 LLM；组件在 __init__ 中创建（无外部依赖即可创建者）"""

    def __init__(self, llm: LLMProtocol):
        self._llm = llm
        self.assembler = PromptAssembler()
        self.router = LLMRouter()
        self.fact_checker = FactChecker(llm, temperature=0)
        self.citation_builder = CitationBuilder()

    async def generate(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()

        # ---- 1. 短路判断 ----------------------------------------------------
        if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT:
            ctx.answer = ""
            ctx.sources = []
            ctx.confidence = 0.0
            ctx.is_fallback = True
            ctx.fallback_level = FallbackLevel.NO_ANSWER
            return ctx

        if ctx.retrieval_eval is RetrievalEval.NEED_MORE:
            ctx.fallback_level = FallbackLevel.PARTIAL

        # ---- 2. 上下文组装 --------------------------------------------------
        assembled_context = self.assembler.assemble(
            ctx.reranked,
            max_chars=settings.generation.max_context_chars,
            threshold=settings.generation.dedup_threshold,
        )

        # ---- 3. 模型路由 ----------------------------------------------------
        try:
            route = self.router.route(ctx.intent)
        except Exception:
            logger.warning("LLMRouter 路由/模板加载失败，使用默认模板降级")
            route = self.router.build_fallback_route()

        # ---- 4. 构建 prompt -------------------------------------------------
        prompt = self._build_output_prompt(
            system=route.system_prompt,
            user_template=route.user_template,
            context=assembled_context,
            query=ctx.query,
        )
        ctx.assembled_prompt = prompt

        # ---- 5. LLM 生成 ----------------------------------------------------
        try:
            raw_answer = await self._call_llm(prompt, route.temperature)
        except Exception:
            logger.warning("GenerationLayer LLM 调用失败，返回空回答")
            raw_answer = ""

        # ---- 6. 事实核查 ----------------------------------------------------
        fact_results: list[FactCheckResult] = []
        pass_rate = 1.0
        fact_check_degraded = False
        if (
            settings.generation.fact_check_enabled
            and raw_answer.strip()
            and assembled_context
        ):
            try:
                fact_results, pass_rate, fact_check_degraded = (
                    await self.fact_checker.check(
                        raw_answer, assembled_context
                    )
                )
            except Exception:
                fact_check_degraded = True
                logger.warning("FactChecker 异常，跳过核查")

        # 注入警示标注
        annotated = self.fact_checker.inject_warnings(raw_answer, fact_results)

        # ---- 7. 引用来源 ----------------------------------------------------
        sources = self.citation_builder.build(ctx.reranked)

        # ---- 8. 置信度 ------------------------------------------------------
        avg_rerank = 0.0
        if ctx.reranked:
            avg_rerank = sum(c.rerank_score for c in ctx.reranked) / len(ctx.reranked)
        confidence = self._compute_confidence(
            avg_rerank, pass_rate, fact_check_degraded
        )

        # ---- 9. 写回 ctx ----------------------------------------------------
        ctx.answer = annotated
        ctx.sources = sources
        ctx.confidence = confidence
        ctx.metadata["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return ctx

    async def _call_llm(self, prompt: str, temperature: float) -> str:
        """调用 LLM；预留为未来 stream 扩展点

        TODO: 当前忽略 route.model_tier/model_name，始终使用 self._llm。
        多模型路由实现后需按 tier 选择不同 LLM 实例。
        """
        return (await self._llm.ainvoke(prompt, temperature=temperature)).content

    @staticmethod
    def _build_output_prompt(
        system: str,
        user_template: str,
        context: str,
        query: str,
    ) -> str:
        """填充 {context} / {query} 占位符，拼接 system + user"""
        cfg = settings.generation
        truncated_query = query[:cfg.max_query_chars] if query else ""
        filled = (
            user_template.replace("{context}", context or "（无参考资料）")
            .replace("{query}", truncated_query)
        )
        prompt = system + "\n\n" + filled
        return prompt

    @staticmethod
    def _compute_confidence(
        avg_rerank_score: float,
        fact_check_pass_rate: float,
        fact_check_degraded: bool,
    ) -> float:
        """confidence = 0.6 * rerank_avg + 0.4 * pass_rate（核查异常时 *0.8）"""
        confidence = 0.6 * avg_rerank_score + 0.4 * fact_check_pass_rate
        if fact_check_degraded:
            confidence *= 0.8
        return round(confidence, 4)
