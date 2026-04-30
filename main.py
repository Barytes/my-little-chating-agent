import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from client import DEFAULT_MODEL, get_openai_client, get_http_client
from tools import AVAILABLE_TOOLS, TOOL_FUNCTIONS

app = FastAPI(title="AI Builders Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Serve frontend at root
@app.get("/")
def serve_frontend():
    """Serve the chat frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not found"}


MAX_TURNS = 5
NESTED_AGENT_MODELS = {"supermind-agent-v1"}
NOTES_PREFETCH_MARKERS = (
    "query_my_note",
    "query_my_notes",
    "my notes",
    "my note",
    "personal knowledge base",
    "我的笔记",
    "我的知识库",
    "个人知识库",
    "我之前",
    "我记录",
    "根据我的",
)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Use query_my_notes as a research assistant for the user's personal knowledge base. When the user's question might be answered by their notes, autonomously formulate one or more targeted semantic search queries and call query_my_notes. If the first notes search is too broad, too narrow, or misses an important angle, refine the query and search again based on what you learned.

Use web_search for current public information and read_page to inspect specific web pages. Prefer query_my_notes over web_search for questions about the user's own projects, plans, decisions, notes, preferences, memories, or previously saved knowledge. When answering from notes, mention the relevant note sources when useful and be clear when the notes do not contain enough information."""


def get_local_agent_model(model: str) -> str:
    """Avoid nesting the Student Portal's own agent inside our local agent loop."""
    if model in NESTED_AGENT_MODELS:
        return DEFAULT_MODEL
    return model


def should_prefetch_notes(message: str) -> bool:
    """Run a backend notes search when the user explicitly points at personal notes."""
    lower_message = message.lower()
    return any(marker in lower_message for marker in NOTES_PREFETCH_MARKERS)


def build_prefetched_notes_context(message: str) -> dict:
    """Call query_my_notes directly and package the result as model context."""
    notes_result = TOOL_FUNCTIONS["query_my_notes"](query=message, top_k=5)
    return {
        "role": "system",
        "content": (
            "The backend already called query_my_notes for this user message. "
            "Use these personal-note search results when answering. If the result contains an error "
            "or weak matches, say so clearly.\n\n"
            f"{json.dumps(notes_result, ensure_ascii=False)}"
        ),
    }


# === Models Endpoints ===


class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    message: str


class ChatResponse(BaseModel):
    model: str
    response: str


class ModelInfo(BaseModel):
    id: str
    owned_by: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default: str


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """
    返回 AI Builders Space 中可用的模型列表
    """
    client = get_openai_client()

    try:
        models_response = client.models.list()
        models = [
            ModelInfo(id=model.id, owned_by=model.owned_by)
            for model in models_response.data
            if model.id not in NESTED_AGENT_MODELS
        ]
        return ModelsResponse(models=models, default=DEFAULT_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Agent loop: 将 message 传入模型，自动处理 Tool Call，最多循环 5 次
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    client = get_openai_client()
    model = get_local_agent_model(request.model)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if should_prefetch_notes(request.message):
        print("[Agent] Prefetching personal notes context")
        messages.append(build_prefetched_notes_context(request.message))

    messages.append({"role": "user", "content": request.message})

    if model != request.model:
        print(f"[Agent] Replacing nested agent model '{request.model}' with '{model}'")
    print(f"[Agent] Starting agent loop with model: {model}")
    print(f"[Agent] User message: {request.message}")

    for turn in range(MAX_TURNS):
        print(f"[Agent] Turn {turn + 1}/{MAX_TURNS}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )

            message = response.choices[0].message

            # Check if LLM wants to call a tool
            if message.tool_calls:
                # Add the assistant's message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute each tool call
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    print(f"[Agent] Decided to call tool: '{tool_name}'")
                    print(f"[Agent] Tool arguments: {tool_args}")

                    # Execute the tool
                    if tool_name in TOOL_FUNCTIONS:
                        tool_result = TOOL_FUNCTIONS[tool_name](**tool_args)
                        result_str = json.dumps(tool_result, indent=2)
                        if len(result_str) > 500:
                            result_str = result_str[:500] + "..."
                        print(f"[System] Tool Output: '{result_str}'")
                    else:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}
                        print(f"[System] Tool Error: Unknown tool '{tool_name}'")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result),
                    })

                # Continue the loop to get LLM's response after tool execution
                continue

            # No tool calls - LLM gave a final answer
            content = message.content or "No response"
            print(f"[Agent] Final Answer: '{content[:200]}...'")

            return ChatResponse(model=model, response=content)

        except Exception as e:
            print(f"[Agent] Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    # Max turns reached without final answer
    print(f"[Agent] Max turns ({MAX_TURNS}) reached without final answer")
    raise HTTPException(
        status_code=500,
        detail=f"Agent loop exceeded max turns ({MAX_TURNS}) without producing a final answer"
    )


# === Search Endpoint ===


class SearchRequest(BaseModel):
    keywords: list[str]
    max_results: int = 5


class QueryResult(BaseModel):
    keyword: str
    results: list[dict]


class SearchResponse(BaseModel):
    queries: list[QueryResult]
    combined_answer: str | None = None
    errors: list | None = None


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    使用 Tavily 进行网络搜索

    Args:
        keywords: 搜索关键词列表
        max_results: 每个关键词最大返回结果数量（默认5，最大20）

    Returns:
        搜索结果，包含每个关键词的结果和综合答案
    """
    if not request.keywords:
        raise HTTPException(status_code=400, detail="keywords cannot be empty")

    try:
        with get_http_client() as client:
            response = client.post(
                "/search/",
                json={
                    "keywords": request.keywords,
                    "max_results": request.max_results,
                },
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Search API error: {response.text}"
                )

            data = response.json()

            queries = [
                QueryResult(
                    keyword=q.get("keyword", ""),
                    results=q.get("results", []),
                )
                for q in data.get("queries", [])
            ]

            return SearchResponse(
                queries=queries,
                combined_answer=data.get("combined_answer"),
                errors=data.get("errors"),
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
