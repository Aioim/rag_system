"""本地 LLM OpenAI 兼容代理服务 — FastAPI 子应用

启动: python -m model.proxy --port 8080
LangChain 集成: ChatOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
"""

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from model.inference import get_local_llm
from model.proxy.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ModelInfo,
    ModelList,
    Usage,
)

app = FastAPI(
    title="Local LLM Proxy",
    description="OpenAI-compatible API for local llama-cpp-python LLM",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelInfo(id="local-llm")])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    llm = get_local_llm()
    # 将 messages 列表拼接为单个 prompt
    prompt = _messages_to_prompt(request.messages)
    kwargs = request.to_kwargs()
    try:
        content = await llm.ainvoke(prompt, **kwargs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def _messages_to_prompt(messages: list[Message]) -> str:
    """将 chat messages 列表拼接为单轮 prompt

    简化实现（非 chat template）：
    - 单条消息 → 直接返回 content
    - 多条消息 → 用角色标签拼接
    """
    if len(messages) == 1:
        return messages[0].content
    parts: list[str] = []
    for msg in messages:
        role = msg.role
        if role == "system":
            parts.append(f"<|system|>\n{msg.content}\n</|system|>")
        elif role == "user":
            parts.append(f"<|user|>\n{msg.content}\n</|user|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}\n</|assistant|>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )
