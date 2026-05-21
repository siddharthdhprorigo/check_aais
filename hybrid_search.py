import argparse
import os
from dataclasses import dataclass
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorFilterMode, VectorizableTextQuery

from main import CONTENT_VECTOR_FIELD_NAME, _make_tenant_slug, _print_json, _required_env


DEFAULT_SEARCH_FIELDS = ["content", "filename"]
DEFAULT_SELECT_FIELDS = [
    "chunk_id",
    "parent_id",
    "tenant_id",
    "kb_id",
    "file_id",
    "filename",
    "blob_path",
    "chunk_ordinal",
    "page_number",
    "source_type",
    "content",
]


@dataclass(slots=True)
class HybridSearchConfig:
    search_endpoint: str
    index_name_prefix: str = "kb-chunks"
    search_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "HybridSearchConfig":
        return cls(
            search_endpoint=_required_env("AZURE_SEARCH_SERVICE_ENDPOINT"),
            index_name_prefix=_env_or_default(
                "AZURE_SEARCH_INDEX_NAME_PREFIX",
                "kb-chunks",
            ),
            search_api_key=_optional_env("AZURE_SEARCH_API_KEY"),
        )

    def index_name_for_tenant(self, tenant_id: str) -> str:
        return f"{self.index_name_prefix}-{_make_tenant_slug(tenant_id)}"


class HybridSearchService:
    def __init__(
        self,
        config: HybridSearchConfig,
        *,
        tenant_id: str,
        search_client: SearchClient | None = None,
    ) -> None:
        self.config = config
        self.tenant_id = _validate_identifier("tenant_id", tenant_id)
        self.index_name = self.config.index_name_for_tenant(self.tenant_id)
        credential = (
            AzureKeyCredential(self.config.search_api_key)
            if self.config.search_api_key
            else DefaultAzureCredential()
        )
        self.search_client = search_client or SearchClient(
            endpoint=self.config.search_endpoint,
            index_name=self.index_name,
            credential=credential,
        )

    def search(
        self,
        *,
        query_text: str,
        kb_id: str | None = None,
        top: int = 5,
        vector_weight: float = 1.0,
        exhaustive: bool = False,
        search_fields: list[str] | None = None,
        select_fields: list[str] | None = None,
        raw_filter: str | None = None,
        vector_filter_mode: str = VectorFilterMode.PRE_FILTER,
        include_content: bool = False,
        max_content_chars: int = 300,
    ) -> dict[str, Any]:
        normalized_query = _validate_identifier("query", query_text)
        normalized_kb_id = (
            None if kb_id is None else _validate_identifier("kb_id", kb_id)
        )
        _validate_positive_int("top", top)
        if vector_weight <= 0:
            raise ValueError("vector_weight must be greater than zero.")
        selected_fields = select_fields or DEFAULT_SELECT_FIELDS
        filter_expression = _build_filter(
            tenant_id=self.tenant_id,
            kb_id=normalized_kb_id,
            raw_filter=raw_filter,
        )
        vector_query = VectorizableTextQuery(
            text=normalized_query,
            fields=CONTENT_VECTOR_FIELD_NAME,
            k_nearest_neighbors=top,
            weight=vector_weight,
            exhaustive=exhaustive,
        )
        results = self.search_client.search(
            search_text=normalized_query,
            search_fields=search_fields or DEFAULT_SEARCH_FIELDS,
            filter=filter_expression,
            include_total_count=True,
            select=selected_fields,
            top=top,
            vector_queries=[vector_query],
            vector_filter_mode=vector_filter_mode,
        )

        items = [
            _format_result(
                rank=index,
                result=result,
                include_content=include_content,
                max_content_chars=max_content_chars,
            )
            for index, result in enumerate(results, start=1)
        ]
        return {
            "query": normalized_query,
            "tenant_id": self.tenant_id,
            "kb_id": normalized_kb_id,
            "index_name": self.index_name,
            "search_fields": search_fields or DEFAULT_SEARCH_FIELDS,
            "select_fields": selected_fields,
            "top": top,
            "vector_top_k": top,
            "vector_weight": vector_weight,
            "vector_filter_mode": vector_filter_mode,
            "filter": filter_expression,
            "count": results.get_count(),
            "results": items,
        }


def _format_result(
    *,
    rank: int,
    result: dict[str, Any],
    include_content: bool,
    max_content_chars: int,
) -> dict[str, Any]:
    payload = dict(result)
    content = payload.get("content")
    formatted = {
        "rank": rank,
        "score": payload.pop("@search.score", None),
        "reranker_score": payload.pop("@search.reranker_score", None),
    }
    formatted.update(payload)
    if isinstance(content, str) and not include_content:
        formatted["content_preview"] = _truncate_text(content, max_content_chars)
        formatted.pop("content", None)
    return formatted


def _build_filter(
    *, tenant_id: str, kb_id: str | None, raw_filter: str | None
) -> str | None:
    filters = [f"tenant_id eq '{_escape_odata_string(tenant_id)}'"]
    if kb_id:
        filters.append(f"kb_id eq '{_escape_odata_string(kb_id)}'")
    if raw_filter:
        filters.append(f"({raw_filter.strip()})")
    return " and ".join(filters) if filters else None


def _escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return f"{value[:max_chars].rstrip()}..."


def _optional_env(name: str) -> str | None:
    value = _env_or_default(name, "")
    return value or None


def _env_or_default(name: str, default: str) -> str:
    env_value = os.getenv(name)
    if env_value is None:
        return default
    normalized = env_value.strip()
    return normalized or default


def _validate_identifier(name: str, value: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank.")
    return normalized


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _parse_csv_list(value: str | None, *, default: list[str]) -> list[str]:
    if value is None:
        return default
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Comma-separated field lists must contain at least one field.")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hybrid full-text plus vector search against a tenant index."
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--kb-id")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument(
        "--vector-filter-mode",
        choices=[mode.value for mode in VectorFilterMode],
        default=VectorFilterMode.PRE_FILTER,
    )
    parser.add_argument("--search-fields")
    parser.add_argument("--select-fields")
    parser.add_argument("--filter")
    parser.add_argument("--max-content-chars", type=int, default=300)
    parser.add_argument("--include-content", action="store_true")
    parser.add_argument("--exhaustive", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = HybridSearchConfig.from_env()
    service = HybridSearchService(config, tenant_id=args.tenant_id)
    response = service.search(
        query_text=args.query,
        kb_id=args.kb_id,
        top=args.top,
        vector_weight=args.vector_weight,
        exhaustive=args.exhaustive,
        search_fields=_parse_csv_list(
            args.search_fields,
            default=DEFAULT_SEARCH_FIELDS,
        ),
        select_fields=_parse_csv_list(
            args.select_fields,
            default=DEFAULT_SELECT_FIELDS,
        ),
        raw_filter=args.filter,
        vector_filter_mode=args.vector_filter_mode,
        include_content=args.include_content,
        max_content_chars=args.max_content_chars,
    )
    _print_json(response)


if __name__ == "__main__":
    main()
