"""Strategic Information Radar workflow."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from client import RADAR_MODEL, get_openai_client
from tools import read_page, web_search


PROJECT_ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = PROJECT_ROOT / "background.md"
MAX_SEARCH_QUERIES = 5
MAX_ARTICLES = 8
ARTICLE_TEXT_LIMIT = 7000
EmitFn = Callable[[dict[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(emit: EmitFn | None, event: dict[str, Any]) -> None:
    if not emit:
        return

    emit({
        "timestamp": _utc_now(),
        **event,
    })


def _preview(value: str, limit: int = 700) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "...[truncated]"


def _current_year() -> int:
    return datetime.now(timezone.utc).year


def read_background() -> str:
    """Read the strategic background used by the radar workflow."""
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(f"Strategic background file not found: {BACKGROUND_PATH}")

    background = BACKGROUND_PATH.read_text(encoding="utf-8").strip()
    if not background:
        raise ValueError("Strategic background is empty")

    return background


def _extract_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from model output, allowing for accidental fences/text."""
    content = content.strip()
    if not content:
        raise ValueError("Model returned an empty response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced:
            parsed = json.loads(fenced.group(1))
        else:
            decoder = json.JSONDecoder()
            start = content.find("{")
            if start == -1:
                raise
            parsed, _ = decoder.raw_decode(content[start:])

    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")

    return parsed


def _call_json_model(messages: list[dict[str, str]], max_tokens: int = 1200) -> dict[str, Any]:
    """Call the radar model and parse a JSON object from the response."""
    client = get_openai_client()
    kwargs = {
        "model": RADAR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "response_format" not in str(exc).lower():
            raise

        kwargs.pop("response_format")
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content or ""
    return _extract_json_object(content)


def _normalize_area(area: Any) -> dict[str, Any]:
    if isinstance(area, str):
        return {"name": area, "rationale": "", "search_queries": [area]}

    if not isinstance(area, dict):
        return {"name": str(area), "rationale": "", "search_queries": [str(area)]}

    name = str(area.get("name") or area.get("area") or area.get("topic") or "").strip()
    rationale = str(area.get("rationale") or area.get("why") or "").strip()
    raw_queries = area.get("search_queries") or area.get("queries") or area.get("keywords") or []

    if isinstance(raw_queries, str):
        queries = [raw_queries]
    elif isinstance(raw_queries, list):
        queries = [str(query).strip() for query in raw_queries if str(query).strip()]
    else:
        queries = []

    if not name:
        name = queries[0] if queries else "Strategic radar topic"

    if not queries:
        queries = [name]

    return {
        "name": name,
        "rationale": rationale,
        "search_queries": [_freshen_search_query(query) for query in queries[:3]],
    }


def _normalize_article(article: Any, source_query: str | None = None) -> dict[str, str] | None:
    if not isinstance(article, dict):
        return None

    url = str(article.get("url") or article.get("link") or article.get("href") or "").strip()
    if not url.startswith(("http://", "https://")):
        return None

    return {
        "title": str(article.get("title") or article.get("name") or url).strip(),
        "url": url,
        "source": str(article.get("source") or article.get("site") or "").strip(),
        "snippet": str(
            article.get("snippet")
            or article.get("content")
            or article.get("description")
            or article.get("summary")
            or ""
        ).strip(),
        "source_query": source_query or str(article.get("source_query") or "").strip(),
        "why_relevant": str(article.get("why_relevant") or article.get("rationale") or "").strip(),
    }


def _freshen_search_query(query: str) -> str:
    """Avoid stale year-constrained queries that often return empty news results."""
    query = re.sub(r"\s+", " ", query).strip()
    current_year = _current_year()

    def replace_stale_year(match: re.Match[str]) -> str:
        year = int(match.group(0))
        if year < current_year - 1:
            return "latest"
        return match.group(0)

    query = re.sub(r"\b20\d{2}\b", replace_stale_year, query)
    if not re.search(r"\b(news|latest|update|updates|breakthrough|release|launch)\b", query, re.I):
        query = f"{query} latest news"

    return query


def _unique_strings(values: list[str], limit: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return unique


def _fallback_search_queries(areas: list[dict[str, Any]]) -> list[str]:
    area_names = [str(area.get("name", "")).strip() for area in areas if area.get("name")]
    query_candidates = [
        "AI video lip-sync real-time generation breakthrough latest news",
        "AI avatar real-time lip sync video generation latest news",
        "HeyGen real-time avatar lip sync latest news",
        "Synthesia real-time AI video lip sync latest news",
        "Runway video generation lip sync latest news",
        "OpenAI Sora real-time video generation latest news",
    ]
    query_candidates.extend(
        f"{name} video lip sync real-time generation latest news"
        for name in area_names
    )
    return _unique_strings([_freshen_search_query(query) for query in query_candidates], MAX_SEARCH_QUERIES)


def _search_result_articles(search_response: dict[str, Any]) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    for query in search_response.get("queries", []) or []:
        keyword = str(query.get("keyword") or query.get("query") or "").strip()
        for result in query.get("results", []) or []:
            article = _normalize_article(result, source_query=keyword)
            if article:
                articles.append(article)

    for result in search_response.get("results", []) or []:
        article = _normalize_article(result)
        if article:
            articles.append(article)

    def collect_nested(value: Any, source_query: str | None = None) -> None:
        if isinstance(value, dict):
            nested_query = str(
                value.get("keyword")
                or value.get("query")
                or value.get("source_query")
                or source_query
                or ""
            ).strip()
            article = _normalize_article(value, source_query=nested_query)
            if article:
                articles.append(article)

            for child in value.values():
                collect_nested(child, source_query=nested_query or source_query)
        elif isinstance(value, list):
            for child in value:
                collect_nested(child, source_query=source_query)

    collect_nested(search_response)

    combined_answer = str(search_response.get("combined_answer") or "")
    for url in re.findall(r"https?://[^\s)>\]]+", combined_answer):
        article = _normalize_article({
            "url": url.rstrip(".,"),
            "title": url.rstrip(".,"),
            "summary": "URL extracted from the search summary answer.",
        })
        if article:
            articles.append(article)

    return articles


def _dedupe_articles(articles: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for article in articles:
        url = article["url"].split("#", 1)[0]
        if url in seen_urls:
            continue

        seen_urls.add(url)
        article["url"] = url
        deduped.append(article)

        if len(deduped) >= MAX_ARTICLES:
            break

    return deduped


def run_broad_scan(background: str, emit: EmitFn | None = None) -> dict[str, Any]:
    """Generate search areas, perform broad news search, and collect candidate URLs."""
    broad_prompt = f"""
Based on the following strategic background, survey 3-5 relevant areas for a broad news search.
Current year: {_current_year()}

Strategic background:
{background}

Return a single JSON object with this shape:
{{
  "areas": [
    {{
      "name": "area name",
      "rationale": "why this area matters",
      "search_queries": ["specific current-news query", "specific competitor query"]
    }}
  ],
  "notes": "short explanation of the broad scan strategy"
}}

Focus on external news about competitor breakthroughs related to video lip-sync and real-time generation.
Prefer current or recent search queries. Do not hard-code old years unless they are still directly relevant.
""".strip()

    _emit(emit, {
        "type": "llm_call",
        "stage": "broad_scan",
        "label": "Generate search areas",
        "model": RADAR_MODEL,
        "prompt_preview": _preview(broad_prompt),
    })
    llm_report = _call_json_model(
        [
            {
                "role": "system",
                "content": (
                    "You are a strategic intelligence analyst. "
                    "Return only valid JSON and prefer current, concrete search queries."
                ),
            },
            {"role": "user", "content": broad_prompt},
        ],
        max_tokens=1200,
    )
    _emit(emit, {
        "type": "llm_result",
        "stage": "broad_scan",
        "label": "Search areas generated",
        "result": llm_report,
    })

    areas = [_normalize_area(area) for area in llm_report.get("areas", [])][:5]
    if not areas:
        areas = [
            {
                "name": "Video lip-sync competitor breakthroughs",
                "rationale": "Directly tied to the project's stated strategic focus.",
                "search_queries": [
                    "video lip-sync AI competitor breakthrough news",
                    "real-time video generation lip sync AI news",
                ],
            }
        ]

    search_queries: list[str] = []
    for area in areas:
        search_queries.extend(area["search_queries"])
    search_queries = _unique_strings(
        [_freshen_search_query(query) for query in search_queries],
        MAX_SEARCH_QUERIES,
    )

    _emit(emit, {
        "type": "tool_call",
        "stage": "broad_scan",
        "label": "Search external news",
        "tool": "web_search",
        "args": {
            "keywords": search_queries,
            "max_results": 3,
        },
    })
    search_response = web_search(search_queries, max_results=3)
    articles = _dedupe_articles(_search_result_articles(search_response))

    if not articles:
        fallback_queries = _fallback_search_queries(areas)
        _emit(emit, {
            "type": "tool_call",
            "stage": "broad_scan",
            "label": "No articles found; retrying with broader queries",
            "tool": "web_search",
            "args": {
                "keywords": fallback_queries,
                "max_results": 5,
            },
        })
        fallback_response = web_search(fallback_queries, max_results=5)
        fallback_articles = _dedupe_articles(_search_result_articles(fallback_response))
        _emit(emit, {
            "type": "tool_result",
            "stage": "broad_scan",
            "label": f"Fallback search found {len(fallback_articles)} candidate articles",
            "tool": "web_search",
            "summary": {
                "article_count": len(fallback_articles),
                "errors": fallback_response.get("errors"),
            },
            "result_preview": fallback_articles,
        })

        if fallback_articles:
            articles = fallback_articles
            search_queries = fallback_queries
            search_response = fallback_response

    _emit(emit, {
        "type": "tool_result",
        "stage": "broad_scan",
        "label": f"Found {len(articles)} candidate articles",
        "tool": "web_search",
        "summary": {
            "article_count": len(articles),
            "errors": search_response.get("errors"),
        },
        "result_preview": articles,
    })

    return {
        "background": background,
        "model": RADAR_MODEL,
        "generated_at": _utc_now(),
        "areas": areas,
        "search_queries": search_queries,
        "articles": articles,
        "search_errors": search_response.get("errors"),
        "combined_answer": search_response.get("combined_answer"),
        "llm_notes": llm_report.get("notes", ""),
    }


def run_deep_dive(
    background: str,
    article: dict[str, str],
    emit: EmitFn | None = None,
) -> dict[str, Any]:
    """Read one article URL and evaluate its strategic importance."""
    _emit(emit, {
        "type": "tool_call",
        "stage": "deep_dive",
        "label": "Read article",
        "tool": "read_page",
        "args": {
            "url": article["url"],
        },
    })
    page = read_page(article["url"])
    if page.get("error"):
        _emit(emit, {
            "type": "tool_result",
            "stage": "deep_dive",
            "label": "Article read failed",
            "tool": "read_page",
            "summary": {
                "url": article["url"],
                "error": page["error"],
            },
        })
        return {
            "url": article["url"],
            "title": article.get("title") or page.get("title") or "",
            "importance": "Low",
            "summary": "The article could not be read.",
            "reasoning": page["error"],
            "read_error": page["error"],
        }

    article_text = str(page.get("text") or "")[:ARTICLE_TEXT_LIMIT]
    title = str(page.get("title") or article.get("title") or article["url"])
    _emit(emit, {
        "type": "tool_result",
        "stage": "deep_dive",
        "label": "Article text extracted",
        "tool": "read_page",
        "summary": {
            "url": article["url"],
            "title": title,
            "text_chars": len(article_text),
        },
        "text_preview": _preview(article_text),
    })

    deep_dive_prompt = f"""
Internal Context:
{background}

External Information:
Title: {title}
URL: {article["url"]}

{article_text}

Analytical Task:
Based on the internal context, evaluate the strategic importance of the external information.
Return a single JSON object with the following fields:
- importance: a string, exactly one of "High", "Medium", or "Low"
- summary: a one-sentence summary
- reasoning: a brief explanation for your importance rating
""".strip()

    _emit(emit, {
        "type": "llm_call",
        "stage": "deep_dive",
        "label": "Analyze strategic importance",
        "model": RADAR_MODEL,
        "article_url": article["url"],
        "prompt_preview": _preview(deep_dive_prompt),
    })
    analysis = _call_json_model(
        [
            {
                "role": "system",
                "content": (
                    "You are a strategic intelligence analyst. "
                    "Return only a valid JSON object with importance, summary, and reasoning."
                ),
            },
            {"role": "user", "content": deep_dive_prompt},
        ],
        max_tokens=700,
    )
    _emit(emit, {
        "type": "llm_result",
        "stage": "deep_dive",
        "label": "Article importance analyzed",
        "article_url": article["url"],
        "result": analysis,
    })

    importance = str(analysis.get("importance") or "Low").strip().title()
    if importance not in {"High", "Medium", "Low"}:
        importance = "Low"

    return {
        "url": article["url"],
        "title": title,
        "source": article.get("source", ""),
        "source_query": article.get("source_query", ""),
        "importance": importance,
        "summary": str(analysis.get("summary") or "").strip(),
        "reasoning": str(analysis.get("reasoning") or "").strip(),
    }


def run_scan(emit: EmitFn | None = None) -> dict[str, Any]:
    """Run the full two-stage Strategic Information Radar scan."""
    _emit(emit, {
        "type": "stage_start",
        "stage": "setup",
        "label": "Reading strategic background",
        "path": str(BACKGROUND_PATH),
    })
    background = read_background()
    _emit(emit, {
        "type": "stage_done",
        "stage": "setup",
        "label": "Strategic background loaded",
        "background": background,
    })

    _emit(emit, {
        "type": "stage_start",
        "stage": "broad_scan",
        "label": "Broad scan started",
    })
    broad_scan_report = run_broad_scan(background, emit=emit)
    _emit(emit, {
        "type": "stage_done",
        "stage": "broad_scan",
        "label": "Broad scan complete",
        "summary": {
            "areas": len(broad_scan_report.get("areas", [])),
            "articles": len(broad_scan_report.get("articles", [])),
        },
    })

    _emit(emit, {
        "type": "stage_start",
        "stage": "deep_dive",
        "label": "Deep dive analysis started",
        "summary": {
            "articles": len(broad_scan_report.get("articles", [])),
        },
    })
    deep_dive_report = []
    for index, article in enumerate(broad_scan_report.get("articles", []), start=1):
        _emit(emit, {
            "type": "article_start",
            "stage": "deep_dive",
            "label": f"Analyzing article {index}",
            "article_index": index,
            "article_count": len(broad_scan_report.get("articles", [])),
            "url": article["url"],
            "title": article.get("title", ""),
        })
        result = run_deep_dive(background, article, emit=emit)
        deep_dive_report.append(result)
        _emit(emit, {
            "type": "article_done",
            "stage": "deep_dive",
            "label": f"Article {index} analyzed",
            "article_index": index,
            "importance": result.get("importance"),
            "summary": result.get("summary"),
            "url": result.get("url"),
        })

    _emit(emit, {
        "type": "stage_done",
        "stage": "deep_dive",
        "label": "Deep dive analysis complete",
        "summary": {
            "articles": len(deep_dive_report),
        },
    })

    return {
        "broad_scan_report": broad_scan_report,
        "deep_dive_report": deep_dive_report,
    }
