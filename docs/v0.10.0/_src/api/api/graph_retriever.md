<a name="base"></a>
# Module base

<a name="text_to_sparql"></a>
# Module text\_to\_sparql

<a name="text_to_sparql.Text2SparqlRetriever"></a>
## Text2SparqlRetriever Objects

```python
class Text2SparqlRetriever(BaseGraphRetriever)
```

Graph retriever that uses a pre-trained Bart model to translate natural language questions given in text form to queries in SPARQL format.
The generated SPARQL query is executed on a knowledge graph.

<a name="text_to_sparql.Text2SparqlRetriever.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(knowledge_graph, model_name_or_path, top_k: int = 1)
```

Init the Retriever by providing a knowledge graph and a pre-trained BART model

**Arguments**:

- `knowledge_graph`: An instance of BaseKnowledgeGraph on which to execute SPARQL queries.
- `model_name_or_path`: Name of or path to a pre-trained BartForConditionalGeneration model.
- `top_k`: How many SPARQL queries to generate per text query.

<a name="text_to_sparql.Text2SparqlRetriever.retrieve"></a>
#### retrieve

```python
 | retrieve(query: str, top_k: Optional[int] = None)
```

Translate a text query to SPARQL and execute it on the knowledge graph to retrieve a list of answers

**Arguments**:

- `query`: Text query that shall be translated to SPARQL and then executed on the knowledge graph
- `top_k`: How many SPARQL queries to generate per text query.

<a name="text_to_sparql.Text2SparqlRetriever.format_result"></a>
#### format\_result

```python
 | format_result(result)
```

Generate formatted dictionary output with text answer and additional info

**Arguments**:

- `result`: The result of a SPARQL query as retrieved from the knowledge graph

