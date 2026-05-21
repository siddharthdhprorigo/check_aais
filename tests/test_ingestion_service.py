from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import (
    CONTENT_VECTOR_FIELD_NAME,
    VECTOR_SEARCH_PROFILE_NAME,
    AzureSearchIngestionConfig,
    AzureSearchIngestionService,
    _load_file_map_from_path,
)


class FakeIndexerClient:
    def __init__(self) -> None:
        self.data_sources = {}
        self.skillsets = {}
        self.indexers = {}
        self.ran_indexers = []
        self.status_by_name = {}

    def create_or_update_data_source_connection(self, data_source):
        self.data_sources[data_source.name] = data_source
        return data_source

    def create_or_update_skillset(self, skillset):
        self.skillsets[skillset.name] = skillset
        return skillset

    def create_or_update_indexer(self, indexer):
        self.indexers[indexer.name] = indexer
        return indexer

    def run_indexer(self, indexer_name: str):
        self.ran_indexers.append(indexer_name)

    def get_indexer_status(self, indexer_name: str):
        if indexer_name not in self.status_by_name:
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError("missing")
        return self.status_by_name[indexer_name]


class FakeIndexClient:
    def __init__(self) -> None:
        self.indexes = {}

    def create_or_update_index(self, index):
        self.indexes[index.name] = index
        return index

    def get_index(self, name: str):
        if name not in self.indexes:
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError("missing")
        return self.indexes[name]


class FakeBlobProperties:
    def __init__(self, metadata=None) -> None:
        self.metadata = metadata or {}


class FakeBlobClient:
    def __init__(self, container, blob_name: str) -> None:
        self.container = container
        self.blob_name = blob_name

    def get_blob_properties(self):
        if self.blob_name not in self.container.blobs:
            raise FileNotFoundError(self.blob_name)
        return FakeBlobProperties(metadata=self.container.blobs[self.blob_name].copy())

    def set_blob_metadata(self, metadata):
        if self.blob_name not in self.container.blobs:
            raise FileNotFoundError(self.blob_name)
        self.container.blobs[self.blob_name] = dict(metadata)
        self.container.metadata_updates.append((self.blob_name, dict(metadata)))


class FakeBlobItem:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeBlobContainerClient:
    def __init__(self, blobs=None) -> None:
        self.blobs = blobs or {}
        self.metadata_updates = []

    def list_blobs(self, name_starts_with: str):
        return [
            FakeBlobItem(name)
            for name in sorted(self.blobs)
            if name.startswith(name_starts_with)
        ]

    def get_blob_client(self, blob_name: str):
        return FakeBlobClient(self, blob_name)


def build_service(blob_names=None):
    config = AzureSearchIngestionConfig(
        search_endpoint="https://example.search.windows.net",
        blob_connection_string="UseDevelopmentStorage=true",
        blob_container_name="docs",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="test-api-key",
        azure_openai_embedding_deployment="embeddings",
        azure_openai_embedding_model_name="text-embedding-3-small",
        azure_openai_embedding_dimensions=1536,
    )
    indexer_client = FakeIndexerClient()
    index_client = FakeIndexClient()
    blob_container_client = FakeBlobContainerClient(
        blobs={name: {} for name in (blob_names or [])}
    )
    service = AzureSearchIngestionService(
        config,
        search_indexer_client=indexer_client,
        index_client=index_client,
        blob_container_client=blob_container_client,
        credential=object(),
    )
    return service, config, indexer_client, index_client, blob_container_client


def test_bootstrap_creates_only_shared_tenant_independent_resources():
    service, config, indexer_client, index_client, _ = build_service()

    result = service.bootstrap()

    assert result["status"] == "bootstrapped"
    assert config.shared_data_source_name in indexer_client.data_sources
    assert index_client.indexes == {}
    assert indexer_client.skillsets == {}


def test_start_ingestion_creates_tenant_index_and_scopes_to_kb_prefix():
    service, _, indexer_client, index_client, blob_container_client = build_service(
        blob_names=["tenant-a/kb-b/12345", "tenant-a/kb-b/98765"]
    )

    result = service.start_ingestion(
        "tenant-a",
        "kb-b",
        {"12345": "invoice.pdf", "98765": "report.docx"},
    )

    assert result["status"] == "accepted"
    assert result["blob_prefix"] == "tenant-a/kb-b"
    assert result["index_name"] in index_client.indexes
    assert result["file_count"] == 2
    assert result["pipeline"] == "tenant_unified"
    assert len(result["indexer_names"]) == 1
    assert set(result["indexer_names"]) == set(indexer_client.ran_indexers)

    data_source_queries = {
        name: data_source.container.query
        for name, data_source in indexer_client.data_sources.items()
    }
    assert len(data_source_queries) == 1
    assert all(query == "tenant-a/" for query in data_source_queries.values())

    assert len(indexer_client.skillsets) == 1
    assert blob_container_client.metadata_updates == [
        (
            "tenant-a/kb-b/12345",
            {
                "filename": "invoice.pdf",
                "file_id": "12345",
                "tenant_id": "tenant-a",
                "kb_id": "kb-b",
            },
        ),
        (
            "tenant-a/kb-b/98765",
            {
                "filename": "report.docx",
                "file_id": "98765",
                "tenant_id": "tenant-a",
                "kb_id": "kb-b",
            },
        ),
    ]

    indexer = next(iter(indexer_client.indexers.values()))
    assert indexer.target_index_name == result["index_name"]
    assert indexer.parameters.configuration.allow_skillset_to_read_file_data is True
    tenant_index = index_client.indexes[result["index_name"]]
    vector_field = next(
        field for field in tenant_index.fields if field.name == CONTENT_VECTOR_FIELD_NAME
    )
    assert str(vector_field.type) == "Collection(Edm.Single)"
    assert vector_field.vector_search_dimensions == 1536
    assert vector_field.vector_search_profile_name == VECTOR_SEARCH_PROFILE_NAME
    assert tenant_index.vector_search is not None
    assert tenant_index.vector_search.profiles[0].name == VECTOR_SEARCH_PROFILE_NAME
    vectorizer = tenant_index.vector_search.vectorizers[0]
    assert vectorizer.parameters.api_key == "test-api-key"
    assert vectorizer.parameters.deployment_name == "embeddings"
    assert vectorizer.parameters.model_name == "text-embedding-3-small"

    skillset = next(iter(indexer_client.skillsets.values()))
    assert len(skillset.skills) == 3
    embedding_skill = next(
        skill for skill in skillset.skills if skill.name == "content-embeddings"
    )
    assert embedding_skill.context == "/document/pages/*"
    assert embedding_skill.inputs[0].source == "/document/pages/*"
    selector = skillset.index_projection.selectors[0]
    assert selector.source_context == "/document/pages/*"


def test_start_ingestion_uses_unified_pipeline_for_text_only_kb():
    service, _, indexer_client, index_client, _ = build_service(
        blob_names=["tenant-a/kb-text/11111", "tenant-a/kb-text/22222"]
    )

    result = service.start_ingestion(
        "tenant-a",
        "kb-text",
        {"11111": "notes.txt", "22222": "summary.md"},
    )

    assert result["pipeline"] == "tenant_unified"
    assert result["index_name"] in index_client.indexes
    assert len(result["indexer_names"]) == 1
    indexer = next(iter(indexer_client.indexers.values()))
    assert indexer.target_index_name == result["index_name"]
    assert indexer.parameters.configuration.allow_skillset_to_read_file_data is True
    skillset = next(iter(indexer_client.skillsets.values()))
    embedding_skill = next(
        skill for skill in skillset.skills if skill.name == "content-embeddings"
    )
    assert embedding_skill.context == "/document/pages/*"
    assert embedding_skill.api_key == "test-api-key"
    assert embedding_skill.deployment_name == "embeddings"
    assert embedding_skill.model_name == "text-embedding-3-small"
    assert embedding_skill.inputs[0].source == "/document/pages/*"
    assert embedding_skill.outputs[0].target_name == "chunk_vector"
    assert len(skillset.skills) == 3
    text_selector = skillset.index_projection.selectors[0]
    assert text_selector.source_context == "/document/pages/*"
    vector_mapping = next(
        mapping for mapping in text_selector.mappings if mapping.name == CONTENT_VECTOR_FIELD_NAME
    )
    assert vector_mapping.source == "/document/pages/*/chunk_vector"


def test_second_kb_for_same_tenant_reuses_same_tenant_index():
    service, _, indexer_client, index_client, _ = build_service(
        blob_names=[
            "tenant-a/kb-1/12345",
            "tenant-a/kb-2/67890",
        ]
    )

    first = service.start_ingestion("tenant-a", "kb-1", {"12345": "doc-1.pdf"})
    second = service.start_ingestion("tenant-a", "kb-2", {"67890": "doc-2.docx"})

    assert first["index_name"] == second["index_name"]
    assert len(index_client.indexes) == 1
    assert len(indexer_client.skillsets) == 1
    assert len(indexer_client.indexers) == 1
    assert len(indexer_client.data_sources) == 1


def test_different_tenants_get_different_indexes():
    service, _, _, index_client, _ = build_service(
        blob_names=[
            "tenant-a/kb-1/12345",
            "tenant-b/kb-1/12345",
        ]
    )

    first = service.start_ingestion("tenant-a", "kb-1", {"12345": "tenant-a.pdf"})
    second = service.start_ingestion("tenant-b", "kb-1", {"12345": "tenant-b.pdf"})

    assert first["index_name"] != second["index_name"]
    assert len(index_client.indexes) == 2


def test_start_ingestion_fails_fast_for_incompatible_existing_index_schema():
    service, _, _, index_client, _ = build_service(
        blob_names=["tenant-a/kb-a/12345"]
    )
    incompatible_index = service._build_index("kb-chunks-tenant-a-placeholder")
    incompatible_index.name = service._tenant_resource_names("tenant-a")["index_name"]
    for field in incompatible_index.fields:
        if field.name == "chunk_ordinal":
            field.type = "Edm.Int32"
    index_client.indexes[incompatible_index.name] = incompatible_index

    with pytest.raises(ValueError, match="Delete and recreate index"):
        service.start_ingestion("tenant-a", "kb-a", {"12345": "doc.pdf"})


def test_start_ingestion_fails_fast_for_existing_index_without_vector_search():
    service, _, _, index_client, _ = build_service(
        blob_names=["tenant-a/kb-a/12345"]
    )
    incompatible_index = service._build_index("kb-chunks-tenant-a-placeholder")
    incompatible_index.name = service._tenant_resource_names("tenant-a")["index_name"]
    incompatible_index.vector_search = None
    index_client.indexes[incompatible_index.name] = incompatible_index

    with pytest.raises(ValueError, match="Delete and recreate index"):
        service.start_ingestion("tenant-a", "kb-a", {"12345": "doc.pdf"})


def test_start_ingestion_fails_fast_for_existing_index_with_wrong_vector_dimensions():
    service, _, _, index_client, _ = build_service(
        blob_names=["tenant-a/kb-a/12345"]
    )
    incompatible_index = service._build_index("kb-chunks-tenant-a-placeholder")
    incompatible_index.name = service._tenant_resource_names("tenant-a")["index_name"]
    for field in incompatible_index.fields:
        if field.name == CONTENT_VECTOR_FIELD_NAME:
            field.vector_search_dimensions = 3072
    index_client.indexes[incompatible_index.name] = incompatible_index

    with pytest.raises(ValueError, match="Delete and recreate index"):
        service.start_ingestion("tenant-a", "kb-a", {"12345": "doc.pdf"})


def test_start_ingestion_rejects_blank_identifiers():
    service, _, _, _, _ = build_service()

    with pytest.raises(ValueError, match="tenant_id"):
        service.start_ingestion(" ", "kb-a", {"12345": "doc.pdf"})

    with pytest.raises(ValueError, match="kb_id"):
        service.start_ingestion("tenant-a", " ", {"12345": "doc.pdf"})


def test_start_ingestion_requires_non_empty_file_map():
    service, _, _, _, _ = build_service()

    with pytest.raises(ValueError, match="file_map"):
        service.start_ingestion("tenant-a", "kb-a", {})


def test_start_ingestion_fails_when_file_map_does_not_match_kb_blobs():
    service, _, indexer_client, _, _ = build_service(
        blob_names=[
            "tenant-a/kb-a/12345",
            "tenant-a/kb-a/98765",
        ]
    )

    with pytest.raises(ValueError, match="file_map must exactly match blobs"):
        service.start_ingestion("tenant-a", "kb-a", {"12345": "doc.pdf"})

    assert indexer_client.ran_indexers == []


def test_start_ingestion_uses_unified_pipeline_for_layout_and_text_files():
    service, _, indexer_client, index_client, blob_container_client = build_service(
        blob_names=[
            "tenant-a/kb-a/12345",
            "tenant-a/kb-a/98765",
        ]
    )

    result = service.start_ingestion(
        "tenant-a",
        "kb-a",
        {"12345": "doc.pdf", "98765": "notes.txt"},
    )

    assert result["pipeline"] == "tenant_unified"
    assert result["index_name"] in index_client.indexes
    assert len(result["indexer_names"]) == 1
    assert len(indexer_client.ran_indexers) == 1
    assert len(indexer_client.skillsets) == 1
    assert blob_container_client.metadata_updates == [
        (
            "tenant-a/kb-a/12345",
            {
                "filename": "doc.pdf",
                "file_id": "12345",
                "tenant_id": "tenant-a",
                "kb_id": "kb-a",
            },
        ),
        (
            "tenant-a/kb-a/98765",
            {
                "filename": "notes.txt",
                "file_id": "98765",
                "tenant_id": "tenant-a",
                "kb_id": "kb-a",
            },
        ),
    ]


def test_start_ingestion_accepts_full_supported_extension_set():
    service, _, _, _, _ = build_service(
        blob_names=[
            "tenant-a/kb-all/1",
            "tenant-a/kb-all/2",
            "tenant-a/kb-all/3",
            "tenant-a/kb-all/4",
            "tenant-a/kb-all/5",
        ]
    )

    result = service.start_ingestion(
        "tenant-a",
        "kb-all",
        {
            "1": "doc.pdf",
            "2": "report.docx",
            "3": "page.html",
            "4": "notes.txt",
            "5": "summary.md",
        },
    )

    assert result["pipeline"] == "tenant_unified"
    assert result["file_count"] == 5


def test_get_ingestion_status_reads_azure_search_status_only():
    service, _, indexer_client, _, _ = build_service(
        blob_names=["tenant-a/kb-b/12345"]
    )

    start_result = service.start_ingestion("tenant-a", "kb-b", {"12345": "doc.pdf"})
    active_indexer_name = start_result["indexer_names"][0]
    indexer_client.status_by_name[active_indexer_name] = SimpleNamespace(
        status="inProgress",
        last_result=SimpleNamespace(
            status="inProgress",
            start_time=None,
            end_time=None,
            errors=[],
            warnings=[],
            items_processed=3,
            items_failed=0,
        ),
        execution_history=[],
    )

    result = service.get_ingestion_status("tenant-a", "kb-b")

    assert result["tenant_id"] == "tenant-a"
    assert result["kb_id"] == "kb-b"
    assert result["index_name"] == start_result["index_name"]
    assert result["status"] == "inProgress"
    found = {
        item["indexer_name"]: item["status"] for item in result["indexers"]
    }
    assert found[active_indexer_name] == "inProgress"
    assert list(found.values()).count("not_found") == 0


def test_get_ingestion_status_reports_missing_indexers():
    service, _, _, _, _ = build_service()

    result = service.get_ingestion_status("tenant-a", "kb-b")

    assert result["status"] == "not_found"
    assert len(result["indexers"]) == 1
    assert all(item["status"] == "not_found" for item in result["indexers"])


def test_load_file_map_from_path(tmp_path):
    file_map_path = tmp_path / "file_map.json"
    file_map_path.write_text('{"12345": "invoice.pdf", "98765": "notes.txt"}', encoding="utf-8")

    loaded = _load_file_map_from_path(str(file_map_path))

    assert loaded == {"12345": "invoice.pdf", "98765": "notes.txt"}


def test_azure_openai_api_key_is_optional_for_vectorizer_and_embedding_skill():
    config = AzureSearchIngestionConfig(
        search_endpoint="https://example.search.windows.net",
        blob_connection_string="UseDevelopmentStorage=true",
        blob_container_name="docs",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key=None,
        azure_openai_embedding_deployment="embeddings",
        azure_openai_embedding_model_name="text-embedding-3-small",
        azure_openai_embedding_dimensions=1536,
    )
    service = AzureSearchIngestionService(
        config,
        search_indexer_client=FakeIndexerClient(),
        index_client=FakeIndexClient(),
        blob_container_client=FakeBlobContainerClient(),
        credential=object(),
    )

    vectorizer = service._build_vector_search().vectorizers[0]
    embedding_skill = service._build_embedding_skill(
        "embeddings",
        "/document/pages/*",
        "/document/pages/*",
        "chunk_vector",
    )

    assert vectorizer.parameters.api_key is None
    assert embedding_skill.api_key is None
