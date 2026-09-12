"""Load model and search providers from providers.json.

model_providers: OpenAI-compatible LLM backends.
search_providers: HTTP search backends used by tools.
The file is re-read when its mtime changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

GLOBAL_ENV = Path.home() / ".config" / "ai-builders" / ".env"


def _ancestor_env_files(start: Path) -> list[Path]:
    files: list[Path] = []
    home = Path.home()
    for path in start.parents:
        if path == home or path.parent == path:
            break
        candidate = path / ".env"
        if candidate.is_file():
            files.append(candidate)
    files.reverse()
    return files


def _load_env() -> None:
    load_dotenv(GLOBAL_ENV)
    here = Path(__file__).resolve().parent
    for env_file in _ancestor_env_files(here):
        load_dotenv(env_file, override=True)
    load_dotenv(here / ".env", override=True)


_load_env()

CONFIG_ENV = "PROVIDERS_CONFIG"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_envs: tuple[str, ...]
    default_model: str
    base_url: str | None = None
    base_url_env: str | None = None
    default_model_env: str | None = None
    allow_empty_key: bool = False
    known_models: tuple[str, ...] = ()

    def resolve_base_url(self) -> str | None:
        if self.base_url_env:
            return os.getenv(self.base_url_env) or self.base_url
        return self.base_url

    def resolve_default_model(self) -> str:
        if self.default_model_env:
            return os.getenv(self.default_model_env) or self.default_model
        return self.default_model


@dataclass(frozen=True)
class SearchProviderConfig:
    name: str
    api_key_envs: tuple[str, ...]
    base_url: str | None = None
    base_url_env: str | None = None
    search_path: str = "/search/"
    allow_empty_key: bool = False

    def resolve_base_url(self) -> str | None:
        if self.base_url_env:
            return os.getenv(self.base_url_env) or self.base_url
        return self.base_url


_raw_cache: dict | None = None
_mtime: float | None = None
_current_search_name: str | None = None


def get_config_path() -> Path:
    override = os.getenv(CONFIG_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "providers.json"


def set_config_path(path: str | Path) -> None:
    os.environ[CONFIG_ENV] = str(Path(path).expanduser().resolve())
    load_config(force=True)


def config_mtime() -> float | None:
    load_config()
    return _mtime


def load_config(force: bool = False) -> dict:
    """Read providers.json. Cached until the file mtime changes."""
    global _raw_cache, _mtime

    path = get_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Provider config not found: {path}")

    mtime = path.stat().st_mtime
    if not force and _raw_cache is not None and _mtime == mtime:
        return _raw_cache

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid provider config: {path} must be an object")
    if not isinstance(data.get("model_providers"), dict) or not data["model_providers"]:
        raise ValueError(f"Invalid provider config: {path} needs a non-empty 'model_providers' object")
    if not isinstance(data.get("search_providers"), dict) or not data["search_providers"]:
        raise ValueError(f"Invalid provider config: {path} needs a non-empty 'search_providers' object")

    _raw_cache = data
    _mtime = mtime
    return data


def reload_providers() -> dict[str, ProviderConfig]:
    load_config(force=True)
    return get_providers()


def _as_tuple(value) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _parse_model_provider(name: str, raw: dict) -> ProviderConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Model provider '{name}' must be an object")
    default_model = raw.get("default_model")
    if not default_model:
        raise ValueError(f"Model provider '{name}' is missing default_model")
    return ProviderConfig(
        name=name,
        api_key_envs=_as_tuple(raw.get("api_key_envs")),
        default_model=str(default_model),
        base_url=raw.get("base_url"),
        base_url_env=raw.get("base_url_env"),
        default_model_env=raw.get("default_model_env"),
        allow_empty_key=bool(raw.get("allow_empty_key", False)),
        known_models=_as_tuple(raw.get("known_models")),
    )


def _parse_search_provider(name: str, raw: dict) -> SearchProviderConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Search provider '{name}' must be an object")
    return SearchProviderConfig(
        name=name,
        api_key_envs=_as_tuple(raw.get("api_key_envs")),
        base_url=raw.get("base_url"),
        base_url_env=raw.get("base_url_env"),
        search_path=str(raw.get("search_path") or "/search/"),
        allow_empty_key=bool(raw.get("allow_empty_key", False)),
    )


def get_providers() -> dict[str, ProviderConfig]:
    data = load_config()
    return {
        name: _parse_model_provider(name, raw)
        for name, raw in data["model_providers"].items()
    }


def get_search_providers() -> dict[str, SearchProviderConfig]:
    data = load_config()
    return {
        name: _parse_search_provider(name, raw)
        for name, raw in data["search_providers"].items()
    }


def get_default_provider_name() -> str:
    env_name = os.getenv("AGENT_PROVIDER")
    if env_name:
        return env_name.strip()
    data = load_config()
    return str(
        data.get("default_model_provider") or next(iter(get_providers()))
    ).strip()


def get_default_search_provider_name() -> str:
    env_name = os.getenv("AGENT_SEARCH_PROVIDER")
    if env_name:
        return env_name.strip()
    data = load_config()
    return str(
        data.get("default_search_provider") or next(iter(get_search_providers()))
    ).strip()


def get_provider(name: str | ProviderConfig | None = None) -> ProviderConfig:
    if isinstance(name, ProviderConfig):
        providers = get_providers()
        return providers.get(name.name, name)

    key = (name or get_default_provider_name()).strip()
    providers = get_providers()
    if key not in providers:
        known = ", ".join(providers)
        raise ValueError(f"Unknown model provider '{name}'. Available: {known}")
    return providers[key]


def get_search_provider(
    name: str | SearchProviderConfig | None = None,
) -> SearchProviderConfig:
    if isinstance(name, SearchProviderConfig):
        providers = get_search_providers()
        return providers.get(name.name, name)

    key = (name or _current_search_name or get_default_search_provider_name()).strip()
    providers = get_search_providers()
    if key not in providers:
        known = ", ".join(providers)
        raise ValueError(f"Unknown search provider '{name}'. Available: {known}")
    return providers[key]


def set_current_search_provider(name: str | None = None) -> SearchProviderConfig:
    global _current_search_name
    cfg = get_search_provider(name or get_default_search_provider_name())
    _current_search_name = cfg.name
    return cfg


def get_current_search_provider() -> SearchProviderConfig:
    return get_search_provider(_current_search_name)


_PLACEHOLDER_KEYS = {
    "your_api_key_here",
    "your-api-key-here",
    "changeme",
    "xxx",
    "todo",
}


def _usable_api_key(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip()
    if not key or key.lower() in _PLACEHOLDER_KEYS:
        return None
    return key


def resolve_api_key(provider: ProviderConfig | SearchProviderConfig) -> str:
    for env_name in provider.api_key_envs:
        api_key = _usable_api_key(os.getenv(env_name))
        if api_key:
            return api_key

    if provider.allow_empty_key:
        return "ollama"

    names = ", ".join(provider.api_key_envs) or "(none)"
    raise ValueError(f"[{provider.name}] API key not configured. Set one of: {names}")


def has_api_key(provider: ProviderConfig | SearchProviderConfig) -> bool:
    if provider.allow_empty_key:
        return True
    return any(_usable_api_key(os.getenv(env_name)) for env_name in provider.api_key_envs)


def validate_provider(provider: ProviderConfig | SearchProviderConfig) -> None:
    resolve_api_key(provider)
    if not provider.resolve_base_url() and getattr(provider, "base_url_env", None):
        raise ValueError(f"[{provider.name}] {provider.base_url_env} is not set")
    if isinstance(provider, SearchProviderConfig) and not provider.resolve_base_url():
        raise ValueError(f"[{provider.name}] search provider is missing base_url")


def list_providers() -> list[ProviderConfig]:
    return list(get_providers().values())


def list_search_providers() -> list[SearchProviderConfig]:
    return list(get_search_providers().values())
