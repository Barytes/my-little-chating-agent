"""Terminal agent: conversation history + tool-calling loop."""

import json

from client import get_llm_client, list_available_models
from providers import (
    ProviderConfig,
    SearchProviderConfig,
    config_mtime,
    get_provider,
    set_current_search_provider,
)
from tools import AVAILABLE_TOOLS, TOOL_FUNCTIONS

MAX_TURNS = 5


class Agent:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        search_provider: str | None = None,
        max_turns: int = MAX_TURNS,
    ):
        self.max_turns = max_turns
        self.messages: list[dict] = []
        self.provider: ProviderConfig
        self.search_provider: SearchProviderConfig
        self.model: str
        self.client = None
        self._config_mtime: float | None = None
        self.set_provider(provider, model)
        self.set_search_provider(search_provider)

    def reset(self) -> None:
        self.messages = []

    def refresh_from_config(self) -> bool:
        """Rebuild clients if providers.json changed. Keep current selections."""
        mtime = config_mtime()
        if mtime == self._config_mtime:
            return False
        self.set_provider(self.provider.name, self.model)
        self.set_search_provider(self.search_provider.name)
        return True

    def set_provider(self, provider: str | None = None, model: str | None = None) -> None:
        cfg = get_provider(provider)
        self.client = get_llm_client(cfg)
        self.provider = cfg
        self.model = model or cfg.resolve_default_model()
        self._config_mtime = config_mtime()

    def set_search_provider(self, provider: str | None = None) -> None:
        self.search_provider = set_current_search_provider(provider)
        self._config_mtime = config_mtime()

    def set_model(self, model: str) -> None:
        name = model.strip()
        if not name:
            raise ValueError("model cannot be empty")
        self.model = name

    def list_models(self) -> tuple[list[str], str]:
        return list_available_models(self.provider, self.client)

    # Agent.chat: Agent loop 核心逻辑
    def chat(self, user_message: str) -> str:
        # 检查用户消息是否为空
        if not user_message:
            raise ValueError("message cannot be empty")

        self.refresh_from_config()
        self.messages.append({"role": "user", "content": user_message})
        print(
            f"[Agent] Model provider: {self.provider.name}  model: {self.model}  "
            f"search: {self.search_provider.name}"
        )
        print(f"[Agent] User message: {user_message}")

        for turn in range(self.max_turns):
            print(f"[Agent] Turn {turn + 1}/{self.max_turns}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )
            message = response.choices[0].message

            if message.tool_calls:
                self._handle_tool_calls(message)
                continue

            content = message.content or "No response"
            self.messages.append({"role": "assistant", "content": content})
            print(f"[Agent] Final Answer: '{content[:200]}...'")
            return content

        raise RuntimeError(
            f"Agent loop exceeded max turns ({self.max_turns}) without a final answer"
        )

    def _handle_tool_calls(self, message) -> None:
        self.messages.append(
            {
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
            }
        )

        for tc in message.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

            print(f"[Agent] Decided to call tool: '{tool_name}'")
            print(f"[Agent] Tool arguments: {tool_args}")

            if tool_name in TOOL_FUNCTIONS:
                tool_result = TOOL_FUNCTIONS[tool_name](**tool_args)
                result_str = json.dumps(tool_result, indent=2)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                print(f"[System] Tool Output: '{result_str}'")
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
                print(f"[System] Tool Error: Unknown tool '{tool_name}'")

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                }
            )
