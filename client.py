import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# AI Builders Space API configuration
BASE_URL = "https://space.ai-builders.com/backend/v1"
API_KEY_ENV_NAMES = ("SUPER_MIND_API_KEY", "BUILDER_API_KEY", "AI_BUILDER_TOKEN")

DEFAULT_MODEL = "grok-4-fast"
MODEL_ALIASES = {
    "grok4fast": "grok-4-fast",
    "grok-4fast": "grok-4-fast",
}
RADAR_MODEL = MODEL_ALIASES.get(
    os.getenv("RADAR_MODEL", DEFAULT_MODEL).strip(),
    os.getenv("RADAR_MODEL", DEFAULT_MODEL).strip(),
)


def get_api_key() -> str:
    """Get API key from environment."""
    for env_name in API_KEY_ENV_NAMES:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key

    names = ", ".join(API_KEY_ENV_NAMES)
    raise ValueError(f"API key not configured. Set one of: {names}")


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured for AI Builders Space."""
    return OpenAI(base_url=BASE_URL, api_key=get_api_key())


def get_http_client() -> httpx.Client:
    """Get HTTP client with authorization header for AI Builders Space."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {get_api_key()}"},
        timeout=30.0,
    )
