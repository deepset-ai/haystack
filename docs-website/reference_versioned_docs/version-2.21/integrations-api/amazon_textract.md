---
title: "Amazon Textract"
id: integrations-amazon_textract
description: "Amazon Textract integration for Haystack"
slug: "/integrations-amazon_textract"
---


## haystack_integrations.components.converters.amazon_textract.converter

### AmazonTextractConverter

Converts documents to Haystack Documents using AWS Textract.

This component uses AWS Textract to extract text and optionally structured data
(tables, forms) from images and single-page PDFs.

When `feature_types` is not set, the component uses `DetectDocumentText` for
plain text OCR. When `feature_types` is set (e.g. `["TABLES", "FORMS"]`), it
uses `AnalyzeDocument` for richer structural analysis.

Natural-language queries are also supported via the `queries` parameter on
`run()`. When queries are provided, the `QUERIES` feature type is added
automatically and Textract returns answers extracted from the document.

Supported input formats: JPEG, PNG, TIFF, BMP, and single-page PDF (up to 10 MB).

AWS credentials are resolved via `Secret` parameters or the default boto3
credential chain (environment variables, AWS config files, IAM roles).

### Usage example

```python
from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

converter = AmazonTextractConverter()
results = converter.run(sources=["document.png"])
documents = results["documents"]
```

#### __init__

```python
__init__(
    *,
    aws_access_key_id: Secret | None = Secret.from_env_var(
        "AWS_ACCESS_KEY_ID", strict=False
    ),
    aws_secret_access_key: Secret | None = Secret.from_env_var(
        "AWS_SECRET_ACCESS_KEY", strict=False
    ),
    aws_session_token: Secret | None = Secret.from_env_var(
        "AWS_SESSION_TOKEN", strict=False
    ),
    aws_region_name: Secret | None = Secret.from_env_var(
        "AWS_DEFAULT_REGION", strict=False
    ),
    aws_profile_name: Secret | None = Secret.from_env_var(
        "AWS_PROFILE", strict=False
    ),
    feature_types: list[str] | None = None,
    store_full_path: bool = False,
    boto3_config: dict[str, Any] | None = None
) -> None
```

Creates an AmazonTextractConverter component.

**Parameters:**

- **aws_access_key_id** (<code>Secret | None</code>) – AWS access key ID.
- **aws_secret_access_key** (<code>Secret | None</code>) – AWS secret access key.
- **aws_session_token** (<code>Secret | None</code>) – AWS session token.
- **aws_region_name** (<code>Secret | None</code>) – AWS region name. Must be a region that supports Textract.
- **aws_profile_name** (<code>Secret | None</code>) – AWS profile name from the credentials file.
- **feature_types** (<code>list\[str\] | None</code>) – List of feature types to detect when using AnalyzeDocument.
  Valid values: "TABLES", "FORMS", "SIGNATURES", "LAYOUT".
  If None, uses DetectDocumentText for basic text extraction.
  The "QUERIES" feature type is managed automatically when the
  `queries` parameter is passed to `run()`.
- **store_full_path** (<code>bool</code>) – If True, stores the complete file path in Document metadata.
  If False, stores only the filename (default).
- **boto3_config** (<code>dict\[str, Any\] | None</code>) – Dictionary of configuration options for the underlying boto3 client.
  Can be used to tune retry behavior, timeouts, and connection management.

#### warm_up

```python
warm_up() -> None
```

Initializes the AWS Textract client.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    queries: list[str] | None = None,
) -> dict[str, Any]
```

Convert documents to Haystack Documents using AWS Textract.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources.
- **queries** (<code>list\[str\] | None</code>) – Optional list of natural-language questions to ask about each document.
  When provided, the Textract `QUERIES` feature type is enabled
  automatically and each question is sent as a query. Answers are
  included in the raw Textract response. Example:
  `["What is the patient name?", "What is the total due?"]`

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of created Documents with extracted text as content.
- `raw_textract_response`: List of raw Textract API responses.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AmazonTextractConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>AmazonTextractConverter</code> – The deserialized component.
