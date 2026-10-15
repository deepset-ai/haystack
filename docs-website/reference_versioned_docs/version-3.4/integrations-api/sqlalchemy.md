---
title: "SQLAlchemy"
id: integrations-sqlalchemy
description: "SQLAlchemy integration for Haystack"
slug: "/integrations-sqlalchemy"
---


## haystack_integrations.components.retrievers.sqlalchemy.sqlalchemy_table_retriever

### SQLAlchemyTableRetriever

Connects to any SQLAlchemy-supported database and executes a SQL query.

Returns results as a Pandas DataFrame and an optional Markdown-formatted table string.
Supports any database backend that SQLAlchemy supports, including PostgreSQL, MySQL,
SQLite, and MSSQL.

### Usage example:

```python
from haystack_integrations.components.retrievers.sqlalchemy import SQLAlchemyTableRetriever

retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
retriever.warm_up()
result = retriever.run(query="SELECT 1 AS value")
print(result["dataframe"])
print(result["table"])
```

#### __init__

```python
__init__(
    drivername: str,
    username: str | None = None,
    password: Secret | None = None,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    init_script: list[str] | None = None,
) -> None
```

Initialize SQLAlchemyTableRetriever.

**Parameters:**

- **drivername** (<code>str</code>) – The SQLAlchemy driver name (e.g., `"sqlite"`,
  `"postgresql+psycopg2"`).
- **username** (<code>str | None</code>) – Database username.
- **password** (<code>Secret | None</code>) – Database password as a Haystack `Secret`.
- **host** (<code>str | None</code>) – Database host.
- **port** (<code>int | None</code>) – Database port.
- **database** (<code>str | None</code>) – Database name or path (e.g., `":memory:"` for SQLite in-memory).
- **init_script** (<code>list\[str\] | None</code>) – Optional list of SQL statements executed once on `warm_up()`
  (e.g., to create tables or insert seed data). Each statement should be a
  separate string in the list.

#### warm_up

```python
warm_up() -> None
```

Initialize the database engine and execute `init_script` if provided.

Called automatically by `run()` on first invocation if not already warmed up.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SQLAlchemyTableRetriever
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SQLAlchemyTableRetriever</code> – Deserialized component.

#### run

```python
run(query: str) -> dict[str, Any]
```

Execute a SQL query and return the results.

**Parameters:**

- **query** (<code>str</code>) – The SQL query to execute.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:

- `dataframe`: A Pandas DataFrame with the query results.

- `table`: A Markdown-formatted string of the results.

- `error`: An error message if the query failed, otherwise an empty string.
