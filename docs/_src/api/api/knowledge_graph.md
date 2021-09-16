<a name="base"></a>
# Module base

<a name="graphdb"></a>
# Module graphdb

<a name="graphdb.GraphDBKnowledgeGraph"></a>
## GraphDBKnowledgeGraph Objects

```python
class GraphDBKnowledgeGraph(BaseKnowledgeGraph)
```

Knowledge graph store that runs on a GraphDB instance

<a name="graphdb.GraphDBKnowledgeGraph.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(host: str = "localhost", port: int = 7200, username: str = "", password: str = "", index: Optional[str] = None, prefixes: str = "")
```

Init the knowledge graph by defining the settings to connect with a GraphDB instance

**Arguments**:

- `host`: address of server where the GraphDB instance is running
- `port`: port where the GraphDB instance is running
- `username`: username to login to the GraphDB instance (if any)
- `password`: password to login to the GraphDB instance (if any)
- `index`: name of the index (also called repository) stored in the GraphDB instance
- `prefixes`: definitions of namespaces with a new line after each namespace, e.g., PREFIX hp: <https://deepset.ai/harry_potter/>

<a name="graphdb.GraphDBKnowledgeGraph.create_index"></a>
#### create\_index

```python
 | create_index(config_path: Path)
```

Create a new index (also called repository) stored in the GraphDB instance

**Arguments**:

- `config_path`: path to a .ttl file with configuration settings, details: https://graphdb.ontotext.com/documentation/free/configuring-a-repository.html#configure-a-repository-programmatically

<a name="graphdb.GraphDBKnowledgeGraph.delete_index"></a>
#### delete\_index

```python
 | delete_index()
```

Delete the index that GraphDBKnowledgeGraph is connected to. This method deletes all data stored in the index.

<a name="graphdb.GraphDBKnowledgeGraph.import_from_ttl_file"></a>
#### import\_from\_ttl\_file

```python
 | import_from_ttl_file(index: str, path: Path)
```

Load an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file into an index of GraphDB

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance where the imported triples shall be stored
- `path`: path to a .ttl containing a knowledge graph

<a name="graphdb.GraphDBKnowledgeGraph.get_all_triples"></a>
#### get\_all\_triples

```python
 | get_all_triples(index: Optional[str] = None)
```

Query the given index in the GraphDB instance for all its stored triples. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance

**Returns**:

all triples stored in the index

<a name="graphdb.GraphDBKnowledgeGraph.get_all_subjects"></a>
#### get\_all\_subjects

```python
 | get_all_subjects(index: Optional[str] = None)
```

Query the given index in the GraphDB instance for all its stored subjects. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance

**Returns**:

all subjects stored in the index

<a name="graphdb.GraphDBKnowledgeGraph.get_all_predicates"></a>
#### get\_all\_predicates

```python
 | get_all_predicates(index: Optional[str] = None)
```

Query the given index in the GraphDB instance for all its stored predicates. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance

**Returns**:

all predicates stored in the index

<a name="graphdb.GraphDBKnowledgeGraph.get_all_objects"></a>
#### get\_all\_objects

```python
 | get_all_objects(index: Optional[str] = None)
```

Query the given index in the GraphDB instance for all its stored objects. Duplicates are not filtered.

**Arguments**:

- `index`: name of the index (also called repository) in the GraphDB instance

**Returns**:

all objects stored in the index

<a name="graphdb.GraphDBKnowledgeGraph.query"></a>
#### query

```python
 | query(sparql_query: str, index: Optional[str] = None)
```

Execute a SPARQL query on the given index in the GraphDB instance

**Arguments**:

- `sparql_query`: SPARQL query that shall be executed
- `index`: name of the index (also called repository) in the GraphDB instance

**Returns**:

query result

