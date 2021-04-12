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

<a name="text_to_sparql.Text2SparqlRetriever.format_result"></a>
#### format\_result

```python
 | format_result(result)
```

Generate formatted dictionary output with text answer and additional info

