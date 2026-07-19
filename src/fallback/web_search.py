"""联网搜索 — DuckDuckGo 实现

使用 ddgs 库进行文本搜索。
搜索失败/超时/未启用时返回 None，不抛异常。
"""
import asyncio

from config import settings
from logger import logger


class WebSearcher:
    """DuckDuckGo 联网搜索器

    封装 ddgs 文本搜索，提供统一接口：
    - 搜索成功 → 返回拼接的搜索结果文本
    - 搜索失败/超时/未启用 → 返回 None
    """

    MAX_RESULTS = 3

    def __init__(self):
        self._DDGS = None  # 懒加载类引用

    @property
    def _ddgs_cls(self):
        """懒加载 DDGS 类引用（缓存后不再重复 import）"""
        if self._DDGS is None:
            from ddgs import DDGS

            self._DDGS = DDGS
        return self._DDGS

    async def search(self, query: str) -> str | None:
        """执行联网搜索

        Args:
            query: 搜索查询字符串

        Returns:
            拼接的搜索结果文本（title + body + href），失败返回 None
        """
        if not settings.web_search.enabled:
            logger.info("联网搜索未启用，跳过")
            return None

        try:
            results = await self._do_search(query)
            if not results:
                logger.info("联网搜索无结果 query=%.100s", query)
                return None

            text = self._format_results(results)
            if text:
                logger.info("联网搜索成功 query=%.100s, results=%d", query, len(results))
            else:
                logger.info("联网搜索结果均无有效内容 query=%.100s, raw_results=%d", query, len(results))
            return text or None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("联网搜索异常 query=%.100s", query)
            return None

    async def _do_search(self, query: str) -> list[dict]:
        """在线程池中执行 ddgs 搜索（同步库，避免阻塞事件循环）"""
        timeout = settings.web_search.timeout_seconds
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_search, query),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("联网搜索超时 query=%.100s, timeout=%ds", query, timeout)
            return []
        except asyncio.CancelledError:
            raise

    def _sync_search(self, query: str) -> list[dict]:
        """同步搜索（在线程池中执行）"""
        with self._ddgs_cls() as ddgs:
            return list(ddgs.text(query, max_results=self.MAX_RESULTS))

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        """将搜索结果格式化为文本"""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            if title == "" and body == "":
                continue
            parts.append(f"[{i}] {title}\n{body}\n来源: {href}")
        return "\n\n".join(parts)
