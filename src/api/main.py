"""RAG API 服务入口（占位 — 后续实现完整 API 路由）"""
from fastapi import FastAPI

app = FastAPI(
    title="RAG Enterprise QA",
    description="企业级 RAG 知识库问答系统中台",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "rag-enterprise-qa"}
