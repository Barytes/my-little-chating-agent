"""Tools for LLM function calling."""

import re

import httpx
from bs4 import BeautifulSoup

from client import get_http_client
from providers import get_current_search_provider

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


def web_search(keywords: list[str], max_results: int = 5) -> dict:
    """
    Perform a web search using Tavily.

    Args:
        keywords: List of search keywords
        max_results: Maximum number of results per keyword (default 5, max 20)

    Returns:
        Search results with queries, combined_answer, and errors
    """
    search = get_current_search_provider()
    with get_http_client() as client:
        response = client.post(
            search.search_path,
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


# Export all available tools and their schemas
AVAILABLE_TOOLS = [WEB_SEARCH_SCHEMA, READ_PAGE_SCHEMA]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_page": read_page,
}