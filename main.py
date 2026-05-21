import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AIServicesAccountKey,
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    DocumentExtractionSkill,
    HnswAlgorithmConfiguration,
    IndexProjectionMode,
    IndexingParameters,
    IndexingParametersConfiguration,
    InputFieldMappingEntry,
    NativeBlobSoftDeleteDeletionDetectionPolicy,
    OutputFieldMappingEntry,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SearchField,
    SearchableField,
    SimpleField,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
    DocumentIntelligenceLayoutSkill,
)
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv

load_dotenv()

LAYOUT_FILE_EXTENSIONS = {".pdf", ".docx", ".html"}
TEXT_FILE_EXTENSIONS = {".txt", ".md"}
CONTENT_VECTOR_FIELD_NAME = "content_vector"
VECTOR_SEARCH_PROFILE_NAME = "content-vector-profile"
VECTOR_SEARCH_ALGORITHM_NAME = "content-vector-hnsw"
VECTOR_SEARCH_VECTORIZER_NAME = "content-vectorizer"


@dataclass(slots=True)
class AzureSearchIngestionConfig:
    search_endpoint: str
    blob_connection_string: str
    blob_container_name: str
    shared_data_source_name: str = "kb-shared-blob-ds"
    index_name_prefix: str = "kb-chunks"
    skillset_name_prefix: str = "kb-skillset"
    indexer_name_prefix: str = "kb-ingestion"
    data_source_name_prefix: str = "kb-ingestion"
    chunk_size: int = 2000
    chunk_overlap: int = 500
    default_language_code: str = "en"
    ai_services_key: str | None = None
    ai_services_subdomain_url: str | None = None
    azure_openai_endpoint: str = ""
    azure_openai_embedding_deployment: str = ""
    azure_openai_embedding_dimensions: int = 0

    @classmethod
    def from_env(cls) -> "AzureSearchIngestionConfig":
        search_endpoint = _required_env("AZURE_SEARCH_SERVICE_ENDPOINT")
        blob_connection_string = _required_env("AZURE_BLOB_CONNECTION_STRING")
        blob_container_name = _required_env("AZURE_BLOB_CONTAINER_NAME")
        azure_openai_endpoint = _required_env("AZURE_OPENAI_ENDPOINT")
        azure_openai_embedding_deployment = _required_env(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )
        azure_openai_embedding_dimensions = int(
            _required_env("AZURE_OPENAI_EMBEDDING_DIMENSIONS")
        )
        chunk_size = int(os.getenv("AZURE_SEARCH_CHUNK_SIZE", "2000"))
        chunk_overlap = int(os.getenv("AZURE_SEARCH_CHUNK_OVERLAP", "500"))
        if chunk_overlap >= (chunk_size / 2):
            raise ValueError(
                "AZURE_SEARCH_CHUNK_OVERLAP must be less than half of "
                "AZURE_SEARCH_CHUNK_SIZE for Azure AI Search chunking."
            )
        return cls(
            search_endpoint=search_endpoint,
            blob_connection_string=blob_connection_string,
            blob_container_name=blob_container_name,
            shared_data_source_name=os.getenv(
                "AZURE_SEARCH_SHARED_DATA_SOURCE_NAME", "kb-shared-blob-ds"
            ),
            index_name_prefix=os.getenv(
                "AZURE_SEARCH_INDEX_NAME_PREFIX", "kb-chunks"
            ),
            skillset_name_prefix=os.getenv(
                "AZURE_SEARCH_SKILLSET_PREFIX", "kb-skillset"
            ),
            indexer_name_prefix=os.getenv(
                "AZURE_SEARCH_INDEXER_PREFIX", "kb-ingestion"
            ),
            data_source_name_prefix=os.getenv(
                "AZURE_SEARCH_DATA_SOURCE_PREFIX", "kb-ingestion"
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            default_language_code=os.getenv(
                "AZURE_SEARCH_DEFAULT_LANGUAGE_CODE", "en"
            ),
            ai_services_key=os.getenv("AZURE_AI_SERVICES_KEY"),
            ai_services_subdomain_url=os.getenv("AZURE_AI_SERVICES_SUBDOMAIN_URL"),
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_embedding_deployment=azure_openai_embedding_deployment,
            azure_openai_embedding_dimensions=azure_openai_embedding_dimensions,
        )


class AzureSearchIngestionService:
    def __init__(
        self,
        config: AzureSearchIngestionConfig,
        *,
        search_indexer_client: SearchIndexerClient | None = None,
        index_client: SearchIndexClient | None = None,
        blob_container_client: ContainerClient | None = None,
        credential: DefaultAzureCredential | None = None,
    ) -> None:
        self.config = config
        self.credential = credential or DefaultAzureCredential()
        self.search_indexer_client = search_indexer_client or SearchIndexerClient(
            self.config.search_endpoint, self.credential
        )
        self.index_client = index_client or SearchIndexClient(
            self.config.search_endpoint, self.credential
        )
        self.blob_container_client = blob_container_client or ContainerClient.from_connection_string(
            self.config.blob_connection_string,
            self.config.blob_container_name,
        )

    def bootstrap(self) -> dict[str, Any]:
        data_source = self._build_shared_data_source()
        self.search_indexer_client.create_or_update_data_source_connection(data_source)

        return {
            "status": "bootstrapped",
            "shared_data_source_name": self.config.shared_data_source_name,
            "tenant_index_name_prefix": self.config.index_name_prefix,
            "tenant_skillset_name_prefix": self.config.skillset_name_prefix,
            "vector_field_name": CONTENT_VECTOR_FIELD_NAME,
            "vector_search_profile_name": VECTOR_SEARCH_PROFILE_NAME,
            "note": (
                "Tenant indexes and tenant-scoped skillsets are created lazily on the "
                "first ingestion for a tenant."
            ),
        }

    def start_ingestion(
        self, tenant_id: str, kb_id: str, file_map: dict[str, str]
    ) -> dict[str, Any]:
        normalized_tenant = _validate_identifier("tenant_id", tenant_id)
        normalized_kb = _validate_identifier("kb_id", kb_id)
        blob_prefix = f"{normalized_tenant}/{normalized_kb}"
        normalized_file_map = _validate_file_map(file_map)
        pipeline = _select_ingestion_pipeline(normalized_file_map)
        tenant_resources = self._tenant_resource_names(normalized_tenant)
        tenant_index_name = tenant_resources["index_name"]
        self._ensure_tenant_index(tenant_index_name)
        active_skillset_name = self._ensure_pipeline_skillset(
            pipeline,
            tenant_resources,
            tenant_index_name,
        )
        self._stamp_blob_metadata(
            normalized_tenant,
            normalized_kb,
            normalized_file_map,
        )
        runtime_resources = self._build_runtime_resources(
            normalized_tenant,
            normalized_kb,
            blob_prefix,
            pipeline=pipeline,
            tenant_index_name=tenant_index_name,
            active_skillset_name=active_skillset_name,
        )

        started_at = _utc_now_iso()
        for resource in runtime_resources:
            self.search_indexer_client.create_or_update_data_source_connection(
                resource["data_source"]
            )
            self.search_indexer_client.create_or_update_indexer(resource["indexer"])
            self.search_indexer_client.run_indexer(resource["indexer"].name)

        return {
            "tenant_id": normalized_tenant,
            "kb_id": normalized_kb,
            "blob_prefix": blob_prefix,
            "index_name": tenant_index_name,
            "file_count": len(normalized_file_map),
            "pipeline": pipeline,
            "indexer_names": [item["indexer"].name for item in runtime_resources],
            "started_at": started_at,
            "status": "accepted",
        }

    def get_ingestion_status(self, tenant_id: str, kb_id: str) -> dict[str, Any]:
        normalized_tenant = _validate_identifier("tenant_id", tenant_id)
        normalized_kb = _validate_identifier("kb_id", kb_id)
        names = self._runtime_resource_names(normalized_tenant, normalized_kb)

        statuses: list[dict[str, Any]] = []
        for doc_group, indexer_name in (
            ("layout", names["layout_indexer_name"]),
            ("text", names["text_indexer_name"]),
            ("mixed", names["mixed_indexer_name"]),
        ):
            statuses.append(
                {
                    "document_group": doc_group,
                    "indexer_name": indexer_name,
                    **self._read_indexer_status(indexer_name),
                }
            )

        return {
            "tenant_id": normalized_tenant,
            "kb_id": normalized_kb,
            "blob_prefix": f"{normalized_tenant}/{normalized_kb}",
            "index_name": self._tenant_resource_names(normalized_tenant)["index_name"],
            "status": _aggregate_status(statuses),
            "indexers": statuses,
        }

    def _build_shared_data_source(self) -> SearchIndexerDataSourceConnection:
        return SearchIndexerDataSourceConnection(
            name=self.config.shared_data_source_name,
            type="azureblob",
            connection_string=self.config.blob_connection_string,
            container=SearchIndexerDataContainer(name=self.config.blob_container_name),
            data_deletion_detection_policy=NativeBlobSoftDeleteDeletionDetectionPolicy(),
        )

    def _build_runtime_resources(
        self,
        tenant_id: str,
        kb_id: str,
        blob_prefix: str,
        *,
        pipeline: str,
        tenant_index_name: str,
        active_skillset_name: str,
    ) -> list[dict[str, Any]]:
        names = self._runtime_resource_names(tenant_id, kb_id)
        if pipeline == "layout":
            return [
                {
                    "kind": "layout",
                    "data_source": self._build_runtime_data_source(
                        names["layout_data_source_name"], blob_prefix
                    ),
                    "indexer": self._build_runtime_indexer(
                        names["layout_indexer_name"],
                        names["layout_data_source_name"],
                        active_skillset_name,
                        tenant_index_name,
                        allow_skillset_to_read_file_data=True,
                    ),
                }
            ]
        if pipeline == "mixed":
            return [
                {
                    "kind": "mixed",
                    "data_source": self._build_runtime_data_source(
                        names["mixed_data_source_name"], blob_prefix
                    ),
                    "indexer": self._build_runtime_indexer(
                        names["mixed_indexer_name"],
                        names["mixed_data_source_name"],
                        active_skillset_name,
                        tenant_index_name,
                        allow_skillset_to_read_file_data=True,
                    ),
                }
            ]
        return [
            {
                "kind": "text",
                "data_source": self._build_runtime_data_source(
                    names["text_data_source_name"], blob_prefix
                ),
                "indexer": self._build_runtime_indexer(
                    names["text_indexer_name"],
                    names["text_data_source_name"],
                    active_skillset_name,
                    tenant_index_name,
                    allow_skillset_to_read_file_data=False,
                ),
            },
        ]

    def _build_runtime_data_source(
        self, data_source_name: str, blob_prefix: str
    ) -> SearchIndexerDataSourceConnection:
        return SearchIndexerDataSourceConnection(
            name=data_source_name,
            type="azureblob",
            connection_string=self.config.blob_connection_string,
            container=SearchIndexerDataContainer(
                name=self.config.blob_container_name,
                query=blob_prefix,
            ),
            data_deletion_detection_policy=NativeBlobSoftDeleteDeletionDetectionPolicy(),
        )

    def _build_runtime_indexer(
        self,
        indexer_name: str,
        data_source_name: str,
        skillset_name: str,
        target_index_name: str,
        *,
        allow_skillset_to_read_file_data: bool,
    ) -> SearchIndexer:
        parameters = IndexingParameters(
            configuration=IndexingParametersConfiguration(
                allow_skillset_to_read_file_data=allow_skillset_to_read_file_data,
            )
        )
        return SearchIndexer(
            name=indexer_name,
            data_source_name=data_source_name,
            target_index_name=target_index_name,
            skillset_name=skillset_name,
            parameters=parameters,
        )

    def _build_index(self, index_name: str) -> SearchIndex:
        fields = [
            SearchableField(
                name="chunk_id",
                type=SearchFieldDataType.STRING,
                key=True,
                filterable=True,
                sortable=True,
                analyzer_name="keyword",
            ),
            SimpleField(
                name="parent_id",
                type=SearchFieldDataType.STRING,
                filterable=True,
            ),
            SimpleField(
                name="tenant_id",
                type=SearchFieldDataType.STRING,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="kb_id",
                type=SearchFieldDataType.STRING,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="file_id",
                type=SearchFieldDataType.STRING,
                filterable=True,
            ),
            SearchableField(
                name="filename",
                type=SearchFieldDataType.STRING,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="blob_path",
                type=SearchFieldDataType.STRING,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.STRING,
            ),
            SearchField(
                name=CONTENT_VECTOR_FIELD_NAME,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.azure_openai_embedding_dimensions,
                vector_search_profile_name=VECTOR_SEARCH_PROFILE_NAME,
            ),
            SimpleField(
                name="chunk_ordinal",
                type=SearchFieldDataType.STRING,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.INT32,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="source_type",
                type=SearchFieldDataType.STRING,
                filterable=True,
                sortable=True,
            ),
        ]
        return SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=self._build_vector_search(),
        )

    def _ensure_tenant_index(self, index_name: str) -> None:
        desired_index = self._build_index(index_name)
        try:
            existing_index = self.index_client.get_index(index_name)
        except ResourceNotFoundError:
            self.index_client.create_or_update_index(desired_index)
            return

        existing_fields = {
            field.name: str(getattr(field, "type", None)) for field in existing_index.fields
        }
        desired_fields = {
            field.name: str(getattr(field, "type", None)) for field in desired_index.fields
        }
        incompatible_fields = [
            field_name
            for field_name, desired_type in desired_fields.items()
            if field_name in existing_fields and existing_fields[field_name] != desired_type
        ]
        incompatible_field_configs = self._find_incompatible_field_configurations(
            existing_index, desired_index
        )
        vector_search_mismatch = self._vector_search_configuration_mismatch(
            existing_index, desired_index
        )
        if incompatible_fields:
            details = ", ".join(
                f"{field_name}: existing={existing_fields[field_name]} desired={desired_fields[field_name]}"
                for field_name in incompatible_fields
            )
            raise ValueError(
                "Tenant index schema is incompatible with the current code. "
                f"Delete and recreate index '{index_name}' before re-running ingestion. "
                f"Incompatible fields: {details}"
            )
        if incompatible_field_configs or vector_search_mismatch:
            details = ", ".join(incompatible_field_configs)
            if vector_search_mismatch:
                details = ", ".join(
                    part for part in [details, "vector_search: existing configuration does not match desired configuration"] if part
                )
            raise ValueError(
                "Tenant index schema is incompatible with the current code. "
                f"Delete and recreate index '{index_name}' before re-running ingestion. "
                f"Incompatible fields: {details}"
            )

        self.index_client.create_or_update_index(desired_index)

    def _ensure_pipeline_skillset(
        self,
        pipeline: str,
        tenant_resources: dict[str, str],
        tenant_index_name: str,
    ) -> str:
        if pipeline == "layout":
            skillset_name = tenant_resources["layout_skillset_name"]
            skillset = self._build_layout_skillset(skillset_name, tenant_index_name)
        elif pipeline == "mixed":
            skillset_name = tenant_resources["mixed_skillset_name"]
            skillset = self._build_mixed_skillset(skillset_name, tenant_index_name)
        else:
            skillset_name = tenant_resources["text_skillset_name"]
            skillset = self._build_text_skillset(skillset_name, tenant_index_name)
        self.search_indexer_client.create_or_update_skillset(skillset)
        return skillset_name

    def _build_layout_skillset(
        self, skillset_name: str, target_index_name: str
    ) -> SearchIndexerSkillset:
        layout_skill = DocumentIntelligenceLayoutSkill(
            name="layout-document-parser",
            description="Extract and chunk PDF/DOCX content with location metadata.",
            context="/document",
            output_mode="oneToMany",
            output_format="text",
            extraction_options=["locationMetadata"],
            chunking_properties={
                "unit": "characters",
                "maximumLength": self.config.chunk_size,
                "overlapLength": self.config.chunk_overlap,
            },
            inputs=[InputFieldMappingEntry(name="file_data", source="/document/file_data")],
            outputs=[
                OutputFieldMappingEntry(name="text_sections", target_name="text_sections")
            ],
        )
        embedding_skill = self._build_embedding_skill(
            name="layout-content-embeddings",
            context="/document/text_sections/*",
            source="/document/text_sections/*/content",
            target_name="chunk_vector",
        )
        return SearchIndexerSkillset(
            name=skillset_name,
            description="Tenant-scoped skillset for PDF/DOCX parsing and chunking.",
            skills=[layout_skill, embedding_skill],
            index_projection=self._build_layout_index_projection(target_index_name),
            cognitive_services_account=self._build_cognitive_services_account(),
        )

    def _build_text_skillset(
        self, skillset_name: str, target_index_name: str
    ) -> SearchIndexerSkillset:
        split_skill = SplitSkill(
            name="split-text-documents",
            description="Chunk TXT/MD/HTML content with overlap.",
            context="/document",
            default_language_code=self.config.default_language_code,
            text_split_mode="pages",
            maximum_page_length=self.config.chunk_size,
            page_overlap_length=self.config.chunk_overlap,
            inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
            outputs=[
                OutputFieldMappingEntry(name="textItems", target_name="pages"),
                OutputFieldMappingEntry(name="ordinalPositions"),
            ],
        )
        embedding_skill = self._build_embedding_skill(
            name="text-content-embeddings",
            context="/document/pages/*",
            source="/document/pages/*",
            target_name="chunk_vector",
        )
        return SearchIndexerSkillset(
            name=skillset_name,
            description="Tenant-scoped skillset for TXT/MD/HTML chunking.",
            skills=[split_skill, embedding_skill],
            index_projection=self._build_text_index_projection(target_index_name),
        )

    def _build_mixed_skillset(
        self, skillset_name: str, target_index_name: str
    ) -> SearchIndexerSkillset:
        extraction_skill = DocumentExtractionSkill(
            name="extract-mixed-document-text",
            description="Extract text from mixed document types before chunking.",
            context="/document",
            data_to_extract="contentAndMetadata",
            inputs=[InputFieldMappingEntry(name="file_data", source="/document/file_data")],
            outputs=[
                OutputFieldMappingEntry(name="content", target_name="extracted_content")
            ],
        )
        split_skill = SplitSkill(
            name="split-mixed-document-text",
            description="Chunk extracted text for mixed document ingestion.",
            context="/document",
            default_language_code=self.config.default_language_code,
            text_split_mode="pages",
            maximum_page_length=self.config.chunk_size,
            page_overlap_length=self.config.chunk_overlap,
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/extracted_content")
            ],
            outputs=[
                OutputFieldMappingEntry(name="textItems", target_name="pages"),
                OutputFieldMappingEntry(name="ordinalPositions"),
            ],
        )
        embedding_skill = self._build_embedding_skill(
            name="mixed-content-embeddings",
            context="/document/pages/*",
            source="/document/pages/*",
            target_name="chunk_vector",
        )
        return SearchIndexerSkillset(
            name=skillset_name,
            description="Tenant-scoped skillset for mixed file type chunking.",
            skills=[extraction_skill, split_skill, embedding_skill],
            index_projection=self._build_text_index_projection(target_index_name),
        )

    def _build_layout_index_projection(
        self, target_index_name: str
    ) -> SearchIndexerIndexProjection:
        return SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=target_index_name,
                    parent_key_field_name="parent_id",
                    source_context="/document/text_sections/*",
                    mappings=[
                        InputFieldMappingEntry(
                            name="content",
                            source="/document/text_sections/*/content",
                        ),
                        InputFieldMappingEntry(
                            name=CONTENT_VECTOR_FIELD_NAME,
                            source="/document/text_sections/*/chunk_vector",
                        ),
                        InputFieldMappingEntry(
                            name="chunk_ordinal",
                            source="/document/text_sections/*/locationMetadata/ordinalPosition",
                        ),
                        InputFieldMappingEntry(
                            name="page_number",
                            source="/document/text_sections/*/locationMetadata/pageNumber",
                        ),
                        InputFieldMappingEntry(
                            name="filename",
                            source="/document/filename",
                        ),
                        InputFieldMappingEntry(
                            name="blob_path",
                            source="/document/metadata_storage_path",
                        ),
                        InputFieldMappingEntry(
                            name="source_type",
                            source="/document/metadata_storage_content_type",
                        ),
                        InputFieldMappingEntry(
                            name="tenant_id",
                            source="/document/tenant_id",
                        ),
                        InputFieldMappingEntry(
                            name="kb_id",
                            source="/document/kb_id",
                        ),
                        InputFieldMappingEntry(
                            name="file_id",
                            source="/document/file_id",
                        ),
                    ],
                )
            ],
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

    def _build_text_index_projection(
        self, target_index_name: str
    ) -> SearchIndexerIndexProjection:
        return SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=target_index_name,
                    parent_key_field_name="parent_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(
                            name="content",
                            source="/document/pages/*",
                        ),
                        InputFieldMappingEntry(
                            name=CONTENT_VECTOR_FIELD_NAME,
                            source="/document/pages/*/chunk_vector",
                        ),
                        InputFieldMappingEntry(
                            name="chunk_ordinal",
                            source="/document/ordinalPositions/*",
                        ),
                        InputFieldMappingEntry(
                            name="filename",
                            source="/document/filename",
                        ),
                        InputFieldMappingEntry(
                            name="blob_path",
                            source="/document/metadata_storage_path",
                        ),
                        InputFieldMappingEntry(
                            name="source_type",
                            source="/document/metadata_storage_content_type",
                        ),
                        InputFieldMappingEntry(
                            name="tenant_id",
                            source="/document/tenant_id",
                        ),
                        InputFieldMappingEntry(
                            name="kb_id",
                            source="/document/kb_id",
                        ),
                        InputFieldMappingEntry(
                            name="file_id",
                            source="/document/file_id",
                        ),
                    ],
                )
            ],
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

    def _build_cognitive_services_account(self) -> AIServicesAccountKey | None:
        if self.config.ai_services_key and self.config.ai_services_subdomain_url:
            return AIServicesAccountKey(
                key=self.config.ai_services_key,
                subdomain_url=self.config.ai_services_subdomain_url,
            )
        return None

    def _build_vector_search(self) -> VectorSearch:
        return VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name=VECTOR_SEARCH_ALGORITHM_NAME),
            ],
            profiles=[
                VectorSearchProfile(
                    name=VECTOR_SEARCH_PROFILE_NAME,
                    algorithm_configuration_name=VECTOR_SEARCH_ALGORITHM_NAME,
                    vectorizer_name=VECTOR_SEARCH_VECTORIZER_NAME,
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name=VECTOR_SEARCH_VECTORIZER_NAME,
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.config.azure_openai_endpoint,
                        deployment_name=self.config.azure_openai_embedding_deployment,
                    ),
                )
            ],
        )

    def _build_embedding_skill(
        self, name: str, context: str, source: str, target_name: str
    ) -> AzureOpenAIEmbeddingSkill:
        return AzureOpenAIEmbeddingSkill(
            name=name,
            context=context,
            resource_url=self.config.azure_openai_endpoint,
            deployment_name=self.config.azure_openai_embedding_deployment,
            dimensions=self.config.azure_openai_embedding_dimensions,
            inputs=[InputFieldMappingEntry(name="text", source=source)],
            outputs=[
                OutputFieldMappingEntry(name="embedding", target_name=target_name)
            ],
        )

    def _find_incompatible_field_configurations(
        self, existing_index: SearchIndex, desired_index: SearchIndex
    ) -> list[str]:
        existing_by_name = {field.name: field for field in existing_index.fields}
        desired_by_name = {field.name: field for field in desired_index.fields}
        mismatches: list[str] = []
        for field_name, desired_field in desired_by_name.items():
            existing_field = existing_by_name.get(field_name)
            if existing_field is None:
                continue
            desired_config = self._normalized_model_data(desired_field)
            existing_config = self._normalized_model_data(existing_field)
            for property_name, desired_value in desired_config.items():
                if property_name in {"name", "type"}:
                    continue
                if property_name not in existing_config:
                    mismatches.append(
                        f"{field_name}: missing property '{property_name}'"
                    )
                    continue
                if existing_config[property_name] != desired_value:
                    mismatches.append(
                        f"{field_name}: existing {property_name}={existing_config[property_name]!r} desired={desired_value!r}"
                    )
        return mismatches

    def _vector_search_configuration_mismatch(
        self, existing_index: SearchIndex, desired_index: SearchIndex
    ) -> bool:
        return self._normalized_model_data(
            getattr(existing_index, "vector_search", None)
        ) != self._normalized_model_data(getattr(desired_index, "vector_search", None))

    def _normalized_model_data(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "_data"):
            return self._normalized_model_data(getattr(value, "_data"))
        if isinstance(value, dict):
            return {
                key: self._normalized_model_data(item)
                for key, item in sorted(value.items())
            }
        if isinstance(value, list):
            return [self._normalized_model_data(item) for item in value]
        return value

    def _stamp_blob_metadata(
        self,
        tenant_id: str,
        kb_id: str,
        file_map: dict[str, str],
    ) -> None:
        prefix = f"{tenant_id}/{kb_id}/"
        blob_items = list(self.blob_container_client.list_blobs(name_starts_with=prefix))
        found_file_ids = {
            _extract_file_id_from_blob_name(getattr(blob, "name", "")) for blob in blob_items
        }
        expected_file_ids = set(file_map.keys())
        if found_file_ids != expected_file_ids:
            missing_in_storage = sorted(expected_file_ids - found_file_ids)
            missing_in_file_map = sorted(found_file_ids - expected_file_ids)
            raise ValueError(
                "file_map must exactly match blobs under the KB prefix. "
                f"missing_in_storage={missing_in_storage}, "
                f"missing_in_file_map={missing_in_file_map}"
            )

        for file_id, file_name in file_map.items():
            blob_name = f"{prefix}{file_id}"
            blob_client = self.blob_container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            metadata = dict(getattr(properties, "metadata", {}) or {})
            metadata.update(
                {
                    "filename": file_name,
                    "file_id": file_id,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                }
            )
            blob_client.set_blob_metadata(metadata=metadata)

    def _tenant_resource_names(self, tenant_id: str) -> dict[str, str]:
        tenant_slug = _make_tenant_slug(tenant_id)
        return {
            "index_name": f"{self.config.index_name_prefix}-{tenant_slug}",
            "layout_skillset_name": (
                f"{self.config.skillset_name_prefix}-layout-{tenant_slug}"
            ),
            "text_skillset_name": (
                f"{self.config.skillset_name_prefix}-text-{tenant_slug}"
            ),
            "mixed_skillset_name": (
                f"{self.config.skillset_name_prefix}-mixed-{tenant_slug}"
            ),
        }

    def _runtime_resource_names(self, tenant_id: str, kb_id: str) -> dict[str, str]:
        slug = _make_resource_slug(tenant_id, kb_id)
        return {
            "layout_data_source_name": (
                f"{self.config.data_source_name_prefix}-layout-{slug}"
            ),
            "text_data_source_name": f"{self.config.data_source_name_prefix}-text-{slug}",
            "mixed_data_source_name": f"{self.config.data_source_name_prefix}-mixed-{slug}",
            "layout_indexer_name": f"{self.config.indexer_name_prefix}-layout-{slug}",
            "text_indexer_name": f"{self.config.indexer_name_prefix}-text-{slug}",
            "mixed_indexer_name": f"{self.config.indexer_name_prefix}-mixed-{slug}",
        }

    def _read_indexer_status(self, indexer_name: str) -> dict[str, Any]:
        try:
            status = self.search_indexer_client.get_indexer_status(indexer_name)
        except ResourceNotFoundError:
            return {
                "status": "not_found",
                "last_result": None,
                "execution_history": [],
            }

        history = []
        for item in getattr(status, "execution_history", []) or []:
            history.append(
                {
                    "status": getattr(item, "status", None),
                    "start_time": _iso_or_none(getattr(item, "start_time", None)),
                    "end_time": _iso_or_none(getattr(item, "end_time", None)),
                }
            )

        last_result = getattr(status, "last_result", None)
        return {
            "status": getattr(status, "status", None),
            "last_result": None
            if last_result is None
            else {
                "status": getattr(last_result, "status", None),
                "start_time": _iso_or_none(getattr(last_result, "start_time", None)),
                "end_time": _iso_or_none(getattr(last_result, "end_time", None)),
                "errors": getattr(last_result, "errors", None),
                "warnings": getattr(last_result, "warnings", None),
                "items_processed": getattr(last_result, "items_processed", None),
                "items_failed": getattr(last_result, "items_failed", None),
            },
            "execution_history": history,
        }


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _validate_identifier(name: str, value: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required.")
    normalized = value.strip().strip("/")
    if not normalized:
        raise ValueError(f"{name} must not be blank.")
    return normalized


def _validate_file_map(file_map: dict[str, str]) -> dict[str, str]:
    if not isinstance(file_map, dict) or not file_map:
        raise ValueError("file_map must be a non-empty dict of file_id to file_name.")
    normalized: dict[str, str] = {}
    for file_id, file_name in file_map.items():
        normalized_file_id = _validate_identifier("file_id", str(file_id))
        normalized_file_name = str(file_name).strip()
        if not normalized_file_name:
            raise ValueError("file_map values must be non-empty file names.")
        normalized[normalized_file_id] = normalized_file_name
    return normalized


def _select_ingestion_pipeline(file_map: dict[str, str]) -> str:
    file_extensions = {
        _extract_file_extension(file_name) for file_name in file_map.values()
    }
    unsupported = sorted(
        ext
        for ext in file_extensions
        if ext not in LAYOUT_FILE_EXTENSIONS and ext not in TEXT_FILE_EXTENSIONS
    )
    if unsupported:
        raise ValueError(
            "Unsupported file extensions in file_map: "
            f"{unsupported}. Supported extensions are "
            f"{sorted(LAYOUT_FILE_EXTENSIONS | TEXT_FILE_EXTENSIONS)}."
        )

    has_layout_files = any(ext in LAYOUT_FILE_EXTENSIONS for ext in file_extensions)
    has_text_files = any(ext in TEXT_FILE_EXTENSIONS for ext in file_extensions)
    if has_layout_files and has_text_files:
        return "mixed"
    return "layout" if has_layout_files else "text"


def _extract_file_id_from_blob_name(blob_name: str) -> str:
    return blob_name.rstrip("/").rsplit("/", 1)[-1]


def _extract_file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def _make_resource_slug(tenant_id: str, kb_id: str) -> str:
    normalized = f"{tenant_id}-{kb_id}".lower()
    safe = re.sub(r"[^a-z0-9-]", "-", normalized)
    safe = re.sub(r"-{2,}", "-", safe).strip("-") or "kb"
    digest = hashlib.sha1(f"{tenant_id}:{kb_id}".encode("utf-8")).hexdigest()[:8]
    base = safe[:40].strip("-") or "kb"
    return f"{base}-{digest}"


def _make_tenant_slug(tenant_id: str) -> str:
    normalized = tenant_id.lower()
    safe = re.sub(r"[^a-z0-9-]", "-", normalized)
    safe = re.sub(r"-{2,}", "-", safe).strip("-") or "tenant"
    digest = hashlib.sha1(tenant_id.encode("utf-8")).hexdigest()[:8]
    base = safe[:40].strip("-") or "tenant"
    return f"{base}-{digest}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _aggregate_status(indexer_statuses: list[dict[str, Any]]) -> str:
    statuses = [item.get("status") for item in indexer_statuses]
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "inProgress" for status in statuses):
        return "inProgress"
    if any(status == "not_found" for status in statuses):
        return "not_found"
    if statuses and all(status == "running" for status in statuses):
        return "running"
    if any(status == "reset" for status in statuses):
        return "reset"
    return "unknown" if not statuses else "completed"


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _load_file_map_from_path(path: str) -> dict[str, str]:
    file_map_path = Path(path)
    with file_map_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return _validate_file_map(loaded)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap and trigger Azure AI Search KB-scoped ingestion."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap", help="Create shared Azure AI Search resources.")

    start_parser = subparsers.add_parser(
        "start", help="Start ingestion for a tenant and knowledge base."
    )
    start_parser.add_argument("--tenant-id", required=True)
    start_parser.add_argument("--kb-id", required=True)
    start_parser.add_argument("--file-map-path", required=True)

    status_parser = subparsers.add_parser(
        "status", help="Get ingestion status for a tenant and knowledge base."
    )
    status_parser.add_argument("--tenant-id", required=True)
    status_parser.add_argument("--kb-id", required=True)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = AzureSearchIngestionConfig.from_env()
    service = AzureSearchIngestionService(config)

    if args.command == "bootstrap":
        _print_json(service.bootstrap())
    elif args.command == "start":
        file_map = _load_file_map_from_path(args.file_map_path)
        _print_json(service.start_ingestion(args.tenant_id, args.kb_id, file_map))
    elif args.command == "status":
        _print_json(service.get_ingestion_status(args.tenant_id, args.kb_id))
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
