import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# AI Builders Space API configuration
BASE_URL = "https://space.ai-builders.com/backend/v1"
API_KEY = os.getenv("BUILDER_API_KEY")

DEFAULT_MODEL = "grok-4-fast"


def get_api_key() -> str:
    """Get API key from environment."""
    if not API_KEY:
        raise ValueError("BUILDER_API_KEY not configured in environment")
    return API_KEY


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