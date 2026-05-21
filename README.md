# Azure AI Search KB-Scoped Ingestion POC

This POC provisions an Azure AI Search ingestion pipeline that keeps parsing, chunking, and vector embedding generation inside Azure AI Search.

It supports KB-scoped ingestion for blobs stored under:

```text
<tenant_id>/<kb_id>/<file_id>
```

where the blob name is the `file_id`, not the original filename.

## Resource model

- One dedicated chunk index per tenant
- One shared blob data source definition
- One tenant-scoped skillset created lazily on first ingestion for that tenant
- One tenant-scoped runtime data source and indexer reused for every KB ingestion request for that tenant
- Blob metadata stamping before indexing so Azure AI Search can ingest `filename`, `file_id`, `tenant_id`, and `kb_id`
- Azure OpenAI-powered vector embeddings persisted per chunk in the tenant index

All KBs for the same tenant land in the same tenant index.

Because blob names are extensionless `file_id` values, the service validates supported file types from the real filenames in `file_map`.

- supported file types: `.pdf`, `.docx`, `.html`, `.txt`, `.md`

## Input contract

Start ingestion with:

- `tenant_id`
- `kb_id`
- `file_map`

`file_map` must be a non-empty mapping shaped like:

```json
{
  "12345": "invoice.pdf",
  "98765": "report.docx"
}
```

The service treats this map as authoritative for the KB ingestion request.

All files in a single request must use supported extensions:

- `.pdf`, `.docx`, `.html`, `.txt`, `.md`

Before Azure AI Search runs, it verifies that the blobs under:

```text
<tenant_id>/<kb_id>/
```

exactly match the provided `file_map` keys, and then stamps each blob with metadata:

- `filename`
- `file_id`
- `tenant_id`
- `kb_id`

## Processing strategy

- All supported file types use one tenant-scoped Azure AI Search built-in pipeline
  - Parsed by `DocumentExtractionSkill`
  - Chunked by `SplitSkill`
  - Embedded by `AzureOpenAIEmbeddingSkill`
  - The tenant indexer scans the tenant prefix and can pick up any changed blobs under that tenant
  - `page_number` and `chunk_ordinal` are nullable and not guaranteed to be populated in this mode

## Vector search readiness

- Every chunk's `content` is embedded during indexing with `AzureOpenAIEmbeddingSkill`
- The embedding is stored in `content_vector` on the tenant index
- The tenant index is provisioned with Azure AI Search vector search configuration and an Azure OpenAI vectorizer
- Downstream applications can use the index for vector similarity or hybrid keyword-plus-vector retrieval in RAG workflows

## Metadata preserved in each tenant index

- `chunk_id`
- `parent_id`
- `tenant_id`
- `kb_id`
- `filename`
- `blob_path`
- `file_id`
- `chunk_ordinal` nullable / best-effort
- `page_number` nullable / best-effort
- `source_type`
- `content_vector`

## Required environment variables

```bash
AZURE_BLOB_CONNECTION_STRING=
AZURE_BLOB_CONTAINER_NAME=
AZURE_SEARCH_SERVICE_ENDPOINT=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_MODEL_NAME=
AZURE_OPENAI_EMBEDDING_DIMENSIONS=
```

## Optional environment variables

```bash
AZURE_SEARCH_SHARED_DATA_SOURCE_NAME=kb-shared-blob-ds
AZURE_SEARCH_INDEX_NAME_PREFIX=kb-chunks
AZURE_SEARCH_SKILLSET_PREFIX=kb-skillset
AZURE_SEARCH_INDEXER_PREFIX=kb-ingestion
AZURE_SEARCH_DATA_SOURCE_PREFIX=kb-ingestion
AZURE_SEARCH_CHUNK_SIZE=2000
AZURE_SEARCH_CHUNK_OVERLAP=500
AZURE_SEARCH_DEFAULT_LANGUAGE_CODE=en
```

`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_MODEL_NAME`, and `AZURE_OPENAI_EMBEDDING_DIMENSIONS` are required so Azure AI Search can generate and persist chunk embeddings in the tenant index.

`AZURE_OPENAI_API_KEY` is optional. When set, Azure AI Search uses that key for the Azure OpenAI vectorizer and embedding skill. When unset, this POC preserves the current non-key auth behavior for those Azure OpenAI calls.

If `AZURE_OPENAI_EMBEDDING_MODEL_NAME` is omitted, this POC falls back to the deployment name for backward compatibility. For newer Azure AI Search API versions, setting the explicit model name is recommended and may be required.

## Commands

Bootstrap shared tenant-independent resources:

```bash
uv run python main.py bootstrap
```

Start ingestion for a KB:

```bash
uv run python main.py start \
  --tenant-id tenant-123 \
  --kb-id kb-456 \
  --file-map-path ./file_map.json
```

Get ingestion status for a KB:

```bash
uv run python main.py status --tenant-id tenant-123 --kb-id kb-456
```

## Python usage

```python
from main import AzureSearchIngestionConfig, AzureSearchIngestionService

config = AzureSearchIngestionConfig.from_env()
service = AzureSearchIngestionService(config)

service.bootstrap()
service.start_ingestion(
    "tenant-123",
    "kb-456",
    {
        "12345": "invoice.pdf",
        "98765": "report.docx",
    },
)
service.get_ingestion_status("tenant-123", "kb-456")
```
