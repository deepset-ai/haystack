<!---
title: "Knowledge Graph"
metaTitle: "Knowledge Graph"
metaDescription: ""
slug: "/docs/knowledgegraph"
date: "2021-04-19"
id: "knowledgegraphmd"
--->

# Question Answering on a Knowledge Graph

Haystack allows loading and querying knowledge graphs. In particular, Haystack can:
 
* Load an existing knowledge graph given as a .ttl file
* Execute SPARQL queries on a knowledge graph
* Execute text queries on the knowledge graph by translating them to SPARQL queries with the help of a pre-trained seq2seq model

Haystack's knowledge graph functionalities are still in a very early stage. Thus, don't expect our [exemplary tutorial](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial10_Knowledge_Graph.py) to work on your custom dataset out-of-the-box.
Two classes implement the functionalities: GraphDBKnowledgeGraph and Text2SparqlRetriever.

## GraphDBKnowledgeGraph

GraphDBKnowledgeGraph is a triple store similar to Haystack's document stores. Currently, it is the only implementation of the BaseKnowledgeGraph class.
GraphDBKnowledgeGraph runs on GraphDB. The licensing of GraphDB is rather complicated and it's more than unfortunate that GraphDB cannot be used right away in colab notebooks.
On your local machine, you can start a GraphDB instance by running:

```docker run -d -p 7200:7200 --name graphdb-instance-tutorial docker-registry.ontotext.com/graphdb-free:9.4.1-adoptopenjdk11```

By default, GraphDBKnowledgeGraph connects to a GraphDB instance running on localhost at port 7200. 
Similar to Haystack's ElasticsearchDocumentStore, the only additional setting needed is an index name.
(Note that GraphDB internally calls these indices repositories.)

```kg = GraphDBKnowledgeGraph(index="tutorial_10_index")```

Indices can be deleted and created with ```GraphDBKnowledgeGraph.delete_index()``` and ```GraphDBKnowledgeGraph.create_index(config_path)```.
```config_path``` needs to point to a .ttl file that contains configuration settings (see [GraphDB documentation](https://graphdb.ontotext.com/documentation/free/configuring-a-repository.html#configure-a-repository-programmatically) for details or use the file from our [tutorial](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial10_Knowledge_Graph.py)). It starts with something like:

```
#
# Sesame configuration template for a GraphDB Free repository
#
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix rep: <http://www.openrdf.org/config/repository#>.
@prefix sr: <http://www.openrdf.org/config/repository/sail#>.
@prefix sail: <http://www.openrdf.org/config/sail#>.
@prefix owlim: <http://www.ontotext.com/trree/owlim#>.

[] a rep:Repository ;
    rep:repositoryID "tutorial_10_index" ;
    rdfs:label "tutorial 10 index" ;
...
```

GraphDBKnowledgeGraph can load an existing knowledge graph represented in the form of a .ttl file with the method ```GraphDBKnowledgeGraph.import_from_ttl_file(index, path)```, where path points to a ttl file starting with something like:

```
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix hp: <https://deepset.ai/harry_potter/> .

hp:Gryffindor hp:source_url "https://harrypotter.fandom.com/wiki/Gryffindor"^^xsd:string .
hp:Gryffindor rdf:type hp:House_ .
hp:Gryffindor hp:name hp:Gryffindor .
hp:Gryffindor hp:founder hp:Godric_gryffindor .
...
```

```GraphDBKnowledgeGraph.get_all_triples()``` returns all loaded triples in the form of subject, predicate, and object. It is helpful to check whether the loading of a .ttl file was successful.

```GraphDBKnowledgeGraph.query(sparql_query)``` executes SPARQL queries on the knowledge graph. However, we usually do not want to use this method directly but use it through a retriever.

## Text2SparqlRetriever
Text2SparqlRetriever can execute SPARQL queries translated from text but also any other custom SPARQL queries. Currently, it is the only implementation of the BaseGraphRetriever class.
Internally, Text2SparqlRetriever uses a pre-trained BART model to translate text questions to queries in SPARQL format.

```Text2SparqlRetriever.retrieve(query)``` can be called with a text query, which is then automatically translated to a SPARQL query.

```Text2SparqlRetriever._query_kg(sparql_query)``` can be called with a SPARQL query.

## Trying Question Answering on Knowledge Graphs with Custom Data
If you want to use your custom data you would first need to have your custom knowledge graph in the format of a .ttl file.
You can load your custom graph and execute SPARQL queries with ```Text2SparqlRetriever._query_kg(sparql_query)```. To allow the use of abbreviations of namespaces, GraphDBKnowledgeGraph needs to know about them:

```
prefixes = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX hp: <https://deepset.ai/harry_potter/>
    """
kg.prefixes = prefixes
```

If you suspect you are having issues because of abbreviations of namespaces not mapped correctly, you can always try to execute a SPARQL query with the full namespace:

```Text2SparqlRetriever._query_kg(sparql_query="select distinct ?obj where { <https://deepset.ai/harry_potter/Hermione_granger> <https://deepset.ai/harry_potter/patronus> ?obj . }")```

instead of using the abbreviated form:

```Text2SparqlRetriever._query_kg(sparql_query="select distinct ?obj where { hp:Hermione_granger hp:patronus ?obj . }")```

If you would like to translate text queries to SPARQL queries for your custom data and use ```Text2SparqlRetriever.retrieve(query)```, there is significantly more effort necessary.
We provide an exemplary pre-trained model in our [tutorial](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial10_Knowledge_Graph.py).
One limitation is that this pre-trained model can only generate questions about resources it has seen during training.
Otherwise, it cannot translate the name of the resource to the identifier used in the knowledge graph.
For example, it can translate "Harry" to "hp:Harry_potter" only because we trained it to do so.

Unfortunately, our pre-trained model for translating text queries does not work with your custom data.
Instead, you need to train your own model. It needs to be trained according to the [seq2seq example for summarization with BART in transformers](https://github.com/huggingface/transformers/tree/master/examples/legacy/seq2seq).
Haystack currently does not support the training of text2sparql models. We dont have concrete plans to extend the funtionality, but we are more than open to contributions. Don't hesitate to reach out! 
