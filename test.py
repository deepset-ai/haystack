from haystack.nodes.retriever import FilterRetriever
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.pipelines import Pipeline

document_store = ElasticsearchDocumentStore(host="localhost", port=9200)
# This threw an error. Now it doesn't
FilterRetriever(document_store).run(root_node="Query", query="")

# This would pass None as query to the nodes because of the same check problem
# Now if the query is "" it passes it to the nodes as it is
p = Pipeline()
p.add_node(FilterRetriever(document_store), name="FilterRetriever", inputs=["Query"])
p.run(query="")
