"""Terminal entry point for the AI agent."""

import argparse
import sys

from .agent import Agent
from .client import list_available_models
from .paths import ensure_runtime_dirs
from .providers import (
    get_config_path,
    get_default_provider_name,
    get_default_search_provider_name,
    get_provider,
    has_api_key,
    list_providers,
    list_search_providers,
    reload_providers,
    set_config_path,
)


def _format_provider_line(cfg, current_name: str) -> str:
    marker = "*" if cfg.name == current_name else " "
    key_state = "key ok" if has_api_key(cfg) else "key missing"
    extra = ""
    if hasattr(cfg, "resolve_default_model"):
        extra = f" default={cfg.resolve_default_model():<16}"
    elif hasattr(cfg, "search_path"):
        extra = f" path={cfg.search_path:<16}"
    return f"  {marker} {cfg.name:<12}{extra} [{key_state}]"


def print_providers(agent: Agent) -> None:
    agent.refresh_from_config()
    print(
        f"当前 model provider: {agent.provider.name}  model: {agent.model}  "
        f"search: {agent.search_provider.name}"
    )
    print("model providers:")
    for cfg in list_providers():
        print(_format_provider_line(cfg, agent.provider.name))
    print("search providers:")
    for cfg in list_search_providers():
        print(_format_provider_line(cfg, agent.search_provider.name))


def handle_provider_command(agent: Agent, user_input: str) -> None:
    parts = user_input.split()
    providers = list_providers()
    if len(parts) == 1:
        print_providers(agent)
        return

    name = resolve_model_choice(parts[1], [cfg.name for cfg in providers])
    model = parts[2] if len(parts) > 2 else None
    agent.set_provider(name, model)
    print(f"已切换 model provider: {agent.provider.name}  model: {agent.model}")


def handle_search_command(agent: Agent, user_input: str) -> None:
    parts = user_input.split()
    providers = list_search_providers()
    if len(parts) == 1:
        print_providers(agent)
        return

    name = resolve_model_choice(parts[1], [cfg.name for cfg in providers])
    agent.set_search_provider(name)
    print(f"已切换 search provider: {agent.search_provider.name}")


def print_models(models: list[str], current: str, source: str, provider_name: str) -> None:
    print(f"当前 provider: {provider_name}  model: {current}")
    print(f"可用模型 ({source}):")
    if not models:
        print("  (空) 可直接 /model 名称 切换")
        return
    for index, name in enumerate(models, start=1):
        marker = "*" if name == current else " "
        print(f"  {marker} {index:<3} {name}")
    print("输入 /model 名称 或 /model 序号 切换")


def resolve_model_choice(choice: str, models: list[str]) -> str:
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(models):
            return models[index - 1]
        raise ValueError(f"序号超出范围: {choice}（1-{len(models)}）")
    return choice


def handle_model_command(agent: Agent, user_input: str) -> None:
    parts = user_input.split(maxsplit=1)
    agent.refresh_from_config()
    models, source = agent.list_models()
    if len(parts) == 1:
        print_models(models, agent.model, source, agent.provider.name)
        return

    name = resolve_model_choice(parts[1].strip(), models)
    if models and name not in models:
        print(f"提示: {name} 不在当前列表中，仍会切换。")
    agent.set_model(name)
    print(f"已切换模型: {agent.model}  (provider: {agent.provider.name})")


def run_repl(agent: Agent) -> None:
    print(
        "终端 Agent。/quit 退出，/clear 清空对话，"
        "/provider [名称|序号] [模型] 切换模型供应商，"
        "/search [名称|序号] 切换搜索供应商，"
        "/model [名称|序号] 切换模型，/reload 重新读取 providers.json。"
    )
    print(f"配置文件: {get_config_path()}")
    print_providers(agent)
    while True:
        try:
            user_input = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "quit", "exit"):
            break
        if user_input == "/clear":
            agent.reset()
            print("对话已清空。")
            continue
        if user_input in ("/reload", "/refresh"):
            try:
                reload_providers()
                agent.refresh_from_config()
                print(f"已重新加载 {get_config_path()}")
                print_providers(agent)
            except Exception as e:
                print(f"错误: {e}")
            continue
        if user_input.startswith("/provider"):
            try:
                handle_provider_command(agent, user_input)
            except Exception as e:
                print(f"错误: {e}")
            continue
        if user_input.startswith("/search"):
            try:
                handle_search_command(agent, user_input)
            except Exception as e:
                print(f"错误: {e}")
            continue
        if user_input.startswith("/model"):
            try:
                handle_model_command(agent, user_input)
            except Exception as e:
                print(f"错误: {e}")
            continue

        try:
            reply = agent.chat(user_input)
        except Exception as e:
            print(f"错误: {e}")
            continue

        print(f"Agent> {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="在终端里和带工具调用的 Agent 对话")
    parser.add_argument("message", nargs="?", help="一次性提问；省略则进入交互模式")
    parser.add_argument(
        "--config",
        default=None,
        help="providers.json 路径，默认读取项目里的 providers.json",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="模型供应商名称或序号；默认用配置文件 / AGENT_PROVIDER",
    )
    parser.add_argument(
        "--search-provider",
        default=None,
        help="搜索供应商名称或序号；默认用配置文件 / AGENT_SEARCH_PROVIDER",
    )
    parser.add_argument("--model", default=None, help="模型名称；默认用该供应商的默认模型")
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="列出可用供应商后退出",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出当前供应商的可用模型后退出",
    )
    args = parser.parse_args()
    ensure_runtime_dirs()

    if args.config:
        set_config_path(args.config)

    provider = args.provider
    if provider and provider.isdigit():
        provider = resolve_model_choice(provider, [cfg.name for cfg in list_providers()])

    search_provider = args.search_provider
    if search_provider and search_provider.isdigit():
        search_provider = resolve_model_choice(
            search_provider, [cfg.name for cfg in list_search_providers()]
        )

    if args.list_providers:
        current_model = get_default_provider_name()
        current_search = get_default_search_provider_name()
        print(f"默认 model provider: {current_model}")
        for cfg in list_providers():
            print(_format_provider_line(cfg, current_model))
        print(f"默认 search provider: {current_search}")
        for cfg in list_search_providers():
            print(_format_provider_line(cfg, current_search))
        return

    if args.list_models:
        try:
            cfg = get_provider(provider)
            models, source = list_available_models(cfg)
            current_model = args.model or cfg.resolve_default_model()
            print_models(models, current_model, source, cfg.name)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        return

    try:
        agent = Agent(
            provider=provider,
            model=args.model,
            search_provider=search_provider,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.message:
        try:
            print(agent.chat(args.message))
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        return

    run_repl(agent)


if __name__ == "__main__":
    main()
