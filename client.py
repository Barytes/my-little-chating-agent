import httpx
from openai import OpenAI

from providers import (
    ProviderConfig,
    get_current_search_provider,
    get_provider,
    resolve_api_key,
    validate_provider,
)


def get_llm_client(provider: str | ProviderConfig | None = None) -> OpenAI:
    """OpenAI-compatible chat client for the given LLM provider."""
    cfg = get_provider(provider)
    validate_provider(cfg)

    kwargs: dict = {"api_key": resolve_api_key(cfg), "base_url": cfg.resolve_base_url()}
    return OpenAI(**kwargs)


_SKIP_MODEL_PREFIXES = (
    "text-embedding",
    "whisper",
    "tts-",
    "dall-e",
    "omni-moderation",
    "babbage",
    "davinci",
)


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(lowered.startswith(prefix) for prefix in _SKIP_MODEL_PREFIXES)


def _catalog_models(cfg: ProviderConfig) -> list[str]:
    models = list(cfg.known_models)
    default = cfg.resolve_default_model()
    if default and default not in models:
        models.insert(0, default)
    return models


def list_available_models(
    provider: str | ProviderConfig | None = None,
    client: OpenAI | None = None,
) -> tuple[list[str], str]:
    """Return (models, source). Prefer live /models, fall back to catalog."""
    cfg = get_provider(provider)
    try:
        llm = client or get_llm_client(cfg)
        remote = [item.id for item in llm.models.list().data if _is_chat_model(item.id)]
        if remote:
            default = cfg.resolve_default_model()
            remote = sorted(set(remote), key=lambda name: (name != default, name))
            return remote, "api"
    except Exception:
        pass
    return _catalog_models(cfg), "catalog"


def get_http_client() -> httpx.Client:
    """Search/tool backend. Uses the current search provider."""
    cfg = get_current_search_provider()
    validate_provider(cfg)
    return httpx.Client(
        base_url=cfg.resolve_base_url(),
        headers={"Authorization": f"Bearer {resolve_api_key(cfg)}"},
        timeout=30.0,
    )
