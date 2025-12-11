---
title: "Snowflake"
id: integrations-snowflake
description: "Snowflake integration for Haystack"
slug: "/integrations-snowflake"
---

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever"></a>

## Module haystack\_integrations.components.retrievers.snowflake.snowflake\_table\_retriever

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever"></a>

### SnowflakeTableRetriever

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
executor.warm_up()
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
executor.warm_up()
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
executor.warm_up()
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

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.__init__"></a>

#### SnowflakeTableRetriever.\_\_init\_\_

```python
def __init__(user: str,
             account: str,
             authenticator: Literal["SNOWFLAKE", "SNOWFLAKE_JWT",
                                    "OAUTH"] = "SNOWFLAKE",
             api_key: Optional[Secret] = Secret.from_env_var(
                 "SNOWFLAKE_API_KEY", strict=False),
             database: Optional[str] = None,
             db_schema: Optional[str] = None,
             warehouse: Optional[str] = None,
             login_timeout: Optional[int] = 60,
             return_markdown: bool = True,
             private_key_file: Optional[Secret] = Secret.from_env_var(
                 "SNOWFLAKE_PRIVATE_KEY_FILE", strict=False),
             private_key_file_pwd: Optional[Secret] = Secret.from_env_var(
                 "SNOWFLAKE_PRIVATE_KEY_PWD", strict=False),
             oauth_client_id: Optional[Secret] = Secret.from_env_var(
                 "SNOWFLAKE_OAUTH_CLIENT_ID", strict=False),
             oauth_client_secret: Optional[Secret] = Secret.from_env_var(
                 "SNOWFLAKE_OAUTH_CLIENT_SECRET", strict=False),
             oauth_token_request_url: Optional[str] = None,
             oauth_authorization_url: Optional[str] = None) -> None
```

**Arguments**:

- `user`: User's login.
- `account`: Snowflake account identifier.
- `authenticator`: Authentication method. Required. Options: "SNOWFLAKE" (password),
"SNOWFLAKE_JWT" (key-pair), or "OAUTH".
- `api_key`: Snowflake account password. Required for SNOWFLAKE authentication.
- `database`: Name of the database to use.
- `db_schema`: Name of the schema to use.
- `warehouse`: Name of the warehouse to use.
- `login_timeout`: Timeout in seconds for login.
- `return_markdown`: Whether to return a Markdown-formatted string of the DataFrame.
- `private_key_file`: Secret containing the path to private key file.
Required for SNOWFLAKE_JWT authentication.
- `private_key_file_pwd`: Secret containing the passphrase for private key file.
Required only when the private key file is encrypted.
- `oauth_client_id`: Secret containing the OAuth client ID.
Required for OAUTH authentication.
- `oauth_client_secret`: Secret containing the OAuth client secret.
Required for OAUTH authentication.
- `oauth_token_request_url`: OAuth token request URL for Client Credentials flow.
- `oauth_authorization_url`: OAuth authorization URL for Authorization Code flow.

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.warm_up"></a>

#### SnowflakeTableRetriever.warm\_up

```python
def warm_up() -> None
```

Warm up the component by initializing the authenticator handler and testing the database connection.

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.to_dict"></a>

#### SnowflakeTableRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.from_dict"></a>

#### SnowflakeTableRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "SnowflakeTableRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.run"></a>

#### SnowflakeTableRetriever.run

```python
@component.output_types(dataframe=DataFrame, table=str)
def run(query: str, return_markdown: Optional[bool] = None) -> Dict[str, Any]
```

Executes a SQL query against a Snowflake database using ADBC and Polars.

**Arguments**:

- `query`: The SQL query to execute.
- `return_markdown`: Whether to return a Markdown-formatted string of the DataFrame.
If not provided, uses the value set during initialization.

**Returns**:

A dictionary containing:
- `"dataframe"`: A Pandas DataFrame with the query results.
- `"table"`: A Markdown-formatted string representation of the DataFrame.
