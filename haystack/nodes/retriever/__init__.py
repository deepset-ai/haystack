from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.dense import DensePassageRetriever, EmbeddingRetriever, TableTextRetriever
from haystack.nodes.retriever.sparse import ElasticsearchRetriever, ElasticsearchFilterOnlyRetriever, TfidfRetriever
from haystack.nodes.retriever.text2sparql import Text2SparqlRetriever