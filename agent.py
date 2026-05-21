import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

from hybrid_search import HybridSearchConfig, HybridSearchService

load_dotenv()


@dataclass(slots=True)
class AgentConfig:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_chat_deployment: str
    openai_api_version: str
    azure_openai_model: str | None = None
    default_top: int = 5

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            azure_openai_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=_required_env("AZURE_OPENAI_API_KEY"),
            azure_openai_chat_deployment=_first_present_env(
                "AZURE_OPENAI_CHAT_DEPLOYMENT",
                "AZURE_OPENAI_DEPLOYMENT",
            ),
            openai_api_version=_env_or_default(
                "AZURE_OPENAI_API_VERSION",
                _env_or_default("OPENAI_API_VERSION", "2024-10-21"),
            ),
            azure_openai_model=_optional_env("AZURE_OPENAI_MODEL"),
            default_top=int(_env_or_default("HYBRID_SEARCH_DEFAULT_TOP", "5")),
        )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip()
    return normalized or default


def _first_present_env(*names: str) -> str:
    for name in names:
        value = _optional_env(name)
        if value:
            return value
    joined = ", ".join(names)
    raise ValueError(f"Missing required environment variable. Set one of: {joined}")


def build_hybrid_search_tool(
    service: HybridSearchService,
    *,
    default_kb_id: str | None,
    default_top: int,
):
    @tool
    def hybrid_search(
        query: str,
        kb_id: str | None = default_kb_id,
        top: int = default_top,
        include_content: bool = True,
    ) -> str:
        """Run hybrid keyword plus vector search over the tenant knowledge base."""
        result = service.search(
            query_text=query,
            kb_id=kb_id,
            top=top,
            include_content=include_content,
        )
        return json.dumps(result, ensure_ascii=True, indent=2)

    return hybrid_search


def build_agent(
    *,
    tenant_id: str,
    kb_id: str | None = None,
    default_top: int | None = None,
):
    agent_config = AgentConfig.from_env()
    search_config = HybridSearchConfig.from_env()
    search_service = HybridSearchService(search_config, tenant_id=tenant_id)
    model = AzureChatOpenAI(
        azure_endpoint=agent_config.azure_openai_endpoint,
        api_key=agent_config.azure_openai_api_key,
        azure_deployment=agent_config.azure_openai_chat_deployment,
        api_version=agent_config.openai_api_version,
        model=agent_config.azure_openai_model,
    )
    search_tool = build_hybrid_search_tool(
        search_service,
        default_kb_id=kb_id,
        default_top=default_top or agent_config.default_top,
    )
    return create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=(
            "You answer questions using the hybrid_search tool over Azure AI Search. "
            "Use the tool whenever the question depends on KB content. "
            "Prefer concise, grounded answers and mention when no relevant results are found."
        ),
    )


def _extract_final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
    return json.dumps(result, ensure_ascii=True, indent=2, default=str)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple LangChain agent backed by Azure OpenAI and hybrid_search."
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--kb-id")
    parser.add_argument("--top", type=int)
    parser.add_argument("--show-trace", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    agent = build_agent(
        tenant_id=args.tenant_id,
        kb_id=args.kb_id,
        default_top=args.top,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    if args.show_trace:
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return
    print(_extract_final_text(result))


if __name__ == "__main__":
    main()
