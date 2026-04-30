"""Tools for LLM function calling."""

import json
import os
import re
from pathlib import Path

import faiss
import httpx
import numpy as np
from bs4 import BeautifulSoup

from client import get_http_client, get_openai_client


PROJECT_ROOT = Path(__file__).resolve().parent
NOTES_ACTIVE_CONFIG_PATH = PROJECT_ROOT / os.getenv(
    "MY_NOTES_ACTIVE_CONFIG_PATH",
    "index/active_notes.json",
)
NOTES_INDEX_PATH = Path(os.getenv("MY_NOTES_INDEX_PATH", "my_notes.index"))
NOTES_METADATA_PATH = Path(os.getenv("MY_NOTES_METADATA_PATH", "my_notes_metadata.json"))
NOTES_EMBEDDING_MODEL = os.getenv("MY_NOTES_EMBEDDING_MODEL", "text-embedding-3-small")

# Function schema for LLM tool calling
WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information using Tavily search engine. Use this tool when you need to find current information, news, facts, or answers to questions that require up-to-date data.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search keywords (e.g., ['Super Bowl 2025', 'winner'])",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return per keyword (default: 5, max: 20)",
                    "default": 5,
                },
            },
            "required": ["keywords"],
        },
    },
}

READ_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": "Fetch a web page and extract its main text content. Use this tool when you need to read the full content of a specific URL found from search results to get detailed information.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to read (e.g., 'https://python.org/downloads/')",
                },
            },
            "required": ["url"],
        },
    },
}

QUERY_MY_NOTES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_my_notes",
        "description": "Search the user's local Markdown notes using the FAISS vector index built from my_notes.index. Use this as a research assistant for the user's personal knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused semantic search query for the user's personal notes.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of matching note chunks to return (default: 5, max: 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

RUN_SCAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_scan",
        "description": (
            "Run the Strategic Information Radar workflow. It reads background.md, "
            "performs a broad news scan, reads candidate article URLs, and returns "
            "a structured strategic importance analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


def web_search(keywords: list[str], max_results: int = 5) -> dict:
    """
    Perform a web search using Tavily.

    Args:
        keywords: List of search keywords
        max_results: Maximum number of results per keyword (default 5, max 20)

    Returns:
        Search results with queries, combined_answer, and errors
    """
    with get_http_client() as client:
        response = client.post(
            "/search/",
            json={
                "keywords": keywords,
                "max_results": max_results,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Search API error: {response.text}")

        return response.json()


def read_page(url: str) -> dict:
    """
    Fetch a web page and extract its main text content.

    Args:
        url: The URL to fetch

    Returns:
        Dictionary with url, title, and text content
    """
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)

            if response.status_code != 200:
                return {
                    "url": url,
                    "error": f"Failed to fetch page: HTTP {response.status_code}",
                }

            html = response.text

            # Parse HTML and extract text
            soup = BeautifulSoup(html, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Get title
            title = soup.title.string if soup.title else ""

            # Get main text
            # Try to find main content area first
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|article|post|entry"))

            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

            # Limit text length to avoid huge responses
            max_length = 5000
            if len(text) > max_length:
                text = text[:max_length] + "...[truncated]"

            return {
                "url": url,
                "title": title.strip() if title else "",
                "text": text,
            }

    except httpx.TimeoutException:
        return {"url": url, "error": "Request timed out"}
    except Exception as e:
        return {"url": url, "error": str(e)}


def _resolve_notes_path(path: Path) -> Path:
    """Resolve note index paths relative to this project if needed."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _get_active_notes_config() -> dict:
    """Return the currently selected notes index config."""
    if NOTES_ACTIVE_CONFIG_PATH.exists():
        return json.loads(NOTES_ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))

    return {
        "notes_name": "default",
        "index_path": str(_resolve_notes_path(NOTES_INDEX_PATH)),
        "metadata_path": str(_resolve_notes_path(NOTES_METADATA_PATH)),
        "embedding_model": NOTES_EMBEDDING_MODEL,
    }


def query_my_notes(query: str, top_k: int = 5) -> dict:
    """
    Search the local Markdown notes FAISS index.

    Args:
        query: Semantic search query
        top_k: Number of matching chunks to return

    Returns:
        Matching note chunks with source metadata and similarity scores
    """
    query = query.strip()
    if not query:
        return {"query": query, "error": "query cannot be empty", "results": []}

    top_k = max(1, min(top_k, 10))
    active_config = _get_active_notes_config()
    index_path = _resolve_notes_path(Path(active_config["index_path"]))
    metadata_path = _resolve_notes_path(Path(active_config["metadata_path"]))
    embedding_model = active_config.get("embedding_model", NOTES_EMBEDDING_MODEL)

    if not index_path.exists():
        return {
            "query": query,
            "error": f"Notes index not found at {index_path}. Run indexer.py first.",
            "results": [],
        }

    if not metadata_path.exists():
        return {
            "query": query,
            "error": f"Notes metadata not found at {metadata_path}. Run indexer.py first.",
            "results": [],
        }

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    client = get_openai_client()
    response = client.embeddings.create(
        model=embedding_model,
        input=query,
    )
    query_vector = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, min(top_k, index.ntotal))
    results = []
    for score, chunk_id in zip(distances[0], indices[0]):
        if chunk_id < 0:
            continue

        item = metadata[chunk_id] if chunk_id < len(metadata) else {}
        results.append({
            "score": float(score),
            "source": item.get("source"),
            "chunk_index": item.get("chunk_index"),
            "start_char": item.get("start_char"),
            "end_char": item.get("end_char"),
            "text": item.get("text"),
        })

    return {
        "query": query,
        "top_k": top_k,
        "notes_name": active_config.get("notes_name"),
        "index_path": str(index_path),
        "results": results,
    }


def run_scan() -> dict:
    """Run the Strategic Information Radar workflow as an agent tool."""
    from radar import run_scan as execute_radar_scan

    return execute_radar_scan()


# Export all available tools and their schemas
AVAILABLE_TOOLS = [
    WEB_SEARCH_SCHEMA,
    READ_PAGE_SCHEMA,
    QUERY_MY_NOTES_SCHEMA,
    RUN_SCAN_SCHEMA,
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_page": read_page,
    "query_my_notes": query_my_notes,
    "run_scan": run_scan,
}
