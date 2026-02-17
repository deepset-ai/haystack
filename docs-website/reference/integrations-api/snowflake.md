---
title: "Snowflake"
id: integrations-snowflake
description: "Snowflake integration for Haystack"
slug: "/integrations-snowflake"
---


## `haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever`

### `SnowflakeTableRetriever`

Connects to a Snowflake database to execute a SQL query using ADBC and Polars.
Returns the results as a Pandas DataFrame (converted from a Polars DataFrame)
along with a Markdown-formatted string.
For more information, see [Polars documentation](https://docs.pola.rs/api/python/dev/reference/api/polars.read_database_uri.html).
and [ADBC documentation](https://arrow.apache.org/adbc/main/driver/snowflake.html).

### Usage examples:

#### Password Authentication:

```python
executor = SnowflakeTableRetriever(
    user="<ACCOUNT-USER>",
    account="<ACCOUNT-IDENTIFIER>",
    authenticator="SNOWFLAKE",
    api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
    database="<DATABASE-NAME>",
    db_schema="<SCHEMA-NAME>",
    warehouse="<WAREHOUSE-NAME>",
)
# Components warm up automatically on first run.
```

#### Key-pair Authentication (MFA):

```python
executor = SnowflakeTableRetriever(
    user="<ACCOUNT-USER>",
    account="<ACCOUNT-IDENTIFIER>",
    authenticator="SNOWFLAKE_JWT",
    private_key_file=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE"),
    private_key_file_pwd=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_PWD"),
    database="<DATABASE-NAME>",
    db_schema="<SCHEMA-NAME>",
    warehouse="<WAREHOUSE-NAME>",
)
# Components warm up automatically on first run.
```

#### OAuth Authentication (MFA):

```python
executor = SnowflakeTableRetriever(
    user="<ACCOUNT-USER>",
    account="<ACCOUNT-IDENTIFIER>",
    authenticator="OAUTH",
    oauth_client_id=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_ID"),
    oauth_client_secret=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_SECRET"),
    oauth_token_request_url="<TOKEN-REQUEST-URL>",
    database="<DATABASE-NAME>",
    db_schema="<SCHEMA-NAME>",
    warehouse="<WAREHOUSE-NAME>",
)
# Components warm up automatically on first run.
```

#### Running queries:

```python
query = "SELECT * FROM table_name"
results = executor.run(query=query)

>> print(results["dataframe"].head(2))

    column1  column2        column3
0     123   'data1'  2024-03-20
1     456   'data2'  2024-03-21

>> print(results["table"])

shape: (3, 3)
| column1 | column2 | column3    |
|---------|---------|------------|
| int     | str     | date       |
|---------|---------|------------|
| 123     | data1   | 2024-03-20 |
| 456     | data2   | 2024-03-21 |
| 789     | data3   | 2024-03-22 |
```

#### `__init__`

```python
__init__(
    user: str,
    account: str,
    authenticator: Literal["SNOWFLAKE", "SNOWFLAKE_JWT", "OAUTH"] = "SNOWFLAKE",
    api_key: Secret | None = Secret.from_env_var(
        "SNOWFLAKE_API_KEY", strict=False
    ),
    database: str | None = None,
    db_schema: str | None = None,
    warehouse: str | None = None,
    login_timeout: int | None = 60,
    return_markdown: bool = True,
    private_key_file: Secret | None = Secret.from_env_var(
        "SNOWFLAKE_PRIVATE_KEY_FILE", strict=False
    ),
    private_key_file_pwd: Secret | None = Secret.from_env_var(
        "SNOWFLAKE_PRIVATE_KEY_PWD", strict=False
    ),
    oauth_client_id: Secret | None = Secret.from_env_var(
        "SNOWFLAKE_OAUTH_CLIENT_ID", strict=False
    ),
    oauth_client_secret: Secret | None = Secret.from_env_var(
        "SNOWFLAKE_OAUTH_CLIENT_SECRET", strict=False
    ),
    oauth_token_request_url: str | None = None,
    oauth_authorization_url: str | None = None,
) -> None
```

**Parameters:**

- **user** (<code>str</code>) – User's login.
- **account** (<code>str</code>) – Snowflake account identifier.
- **authenticator** (<code>Literal['SNOWFLAKE', 'SNOWFLAKE_JWT', 'OAUTH']</code>) – Authentication method. Required. Options: "SNOWFLAKE" (password),
  "SNOWFLAKE_JWT" (key-pair), or "OAUTH".
- **api_key** (<code>Secret | None</code>) – Snowflake account password. Required for SNOWFLAKE authentication.
- **database** (<code>str | None</code>) – Name of the database to use.
- **db_schema** (<code>str | None</code>) – Name of the schema to use.
- **warehouse** (<code>str | None</code>) – Name of the warehouse to use.
- **login_timeout** (<code>int | None</code>) – Timeout in seconds for login.
- **return_markdown** (<code>bool</code>) – Whether to return a Markdown-formatted string of the DataFrame.
- **private_key_file** (<code>Secret | None</code>) – Secret containing the path to private key file.
  Required for SNOWFLAKE_JWT authentication.
- **private_key_file_pwd** (<code>Secret | None</code>) – Secret containing the passphrase for private key file.
  Required only when the private key file is encrypted.
- **oauth_client_id** (<code>Secret | None</code>) – Secret containing the OAuth client ID.
  Required for OAUTH authentication.
- **oauth_client_secret** (<code>Secret | None</code>) – Secret containing the OAuth client secret.
  Required for OAUTH authentication.
- **oauth_token_request_url** (<code>str | None</code>) – OAuth token request URL for Client Credentials flow.
- **oauth_authorization_url** (<code>str | None</code>) – OAuth authorization URL for Authorization Code flow.

#### `warm_up`

```python
warm_up() -> None
```

Warm up the component by initializing the authenticator handler and testing the database connection.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SnowflakeTableRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SnowflakeTableRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str, return_markdown: bool | None = None
) -> dict[str, DataFrame | str]
```

Executes a SQL query against a Snowflake database using ADBC and Polars.

**Parameters:**

- **query** (<code>str</code>) – The SQL query to execute.
- **return_markdown** (<code>bool | None</code>) – Whether to return a Markdown-formatted string of the DataFrame.
  If not provided, uses the value set during initialization.

**Returns:**

- <code>dict\[str, DataFrame | str\]</code> – A dictionary containing:
- `"dataframe"`: A Pandas DataFrame with the query results.
- `"table"`: A Markdown-formatted string representation of the DataFrame.
