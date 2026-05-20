# Azure AI Search KB-Scoped Ingestion POC

This POC provisions an Azure AI Search ingestion pipeline that keeps parsing and chunking inside Azure AI Search.

It supports KB-scoped ingestion for blobs stored under:

```text
<tenant_id>/<kb_id>/<file_id>
```

where the blob name is the `file_id`, not the original filename.

## Resource model

- One dedicated chunk index per tenant
- One shared blob data source definition
- Tenant-scoped skillsets created lazily on first ingestion for that tenant
- KB-scoped runtime data sources and indexers for each ingestion request
- Blob metadata stamping before indexing so Azure AI Search can ingest `filename`, `file_id`, `tenant_id`, and `kb_id`

All KBs for the same tenant land in the same tenant index.

Because blob names are extensionless `file_id` values, the service chooses the Azure AI Search pipeline from the real filenames in `file_map`.

- layout family: `.pdf`, `.docx`, `.html`
- text family: `.txt`, `.md`
- mixed layout and text families use a mixed extraction pipeline in Azure AI Search

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

All files in a single request must belong to the same Azure AI Search pipeline family:

- layout family: `.pdf`, `.docx`, `.html`
- text family: `.txt`, `.md`
- mixed family: any combination across those two sets, handled through `DocumentExtractionSkill -> SplitSkill`

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

- `PDF`, `DOCX`, and `HTML`
  - Parsed by Azure AI Search using `DocumentIntelligenceLayoutSkill`
  - Chunked by Azure AI Search using fixed-size overlap-aware text sections
  - Preserves `page_number` when Azure returns `locationMetadata`

- `TXT` and `MD`
  - Parsed by the blob indexer
  - Chunked by Azure AI Search using `SplitSkill`

- Mixed `PDF`/`DOCX`/`HTML` with `TXT`/`MD`
  - Parsed by Azure AI Search using `DocumentExtractionSkill`
  - Chunked by Azure AI Search using `SplitSkill`
  - Trades away page metadata in favor of supporting mixed file types in one KB ingestion

## Metadata preserved in each tenant index

- `chunk_id`
- `parent_id`
- `tenant_id`
- `kb_id`
- `filename`
- `blob_path`
- `file_id`
- `chunk_ordinal`
- `page_number` when available
- `source_type`

## Required environment variables

```bash
AZURE_BLOB_CONNECTION_STRING=
AZURE_BLOB_CONTAINER_NAME=
AZURE_SEARCH_SERVICE_ENDPOINT=
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
AZURE_AI_SERVICES_KEY=
AZURE_AI_SERVICES_SUBDOMAIN_URL=
```

`AZURE_AI_SERVICES_KEY` and `AZURE_AI_SERVICES_SUBDOMAIN_URL` are only needed when your Document Layout skill billing setup uses an attached Azure AI / Foundry resource key.

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
