from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.dense import (
    DenseRetriever,
    DensePassageRetriever,
    EmbeddingRetriever,
    MultihopEmbeddingRetriever,
    TableTextRetriever,
)
from haystack.nodes.retriever.sparse import (
    BM25Retriever,
    ElasticsearchRetriever,
    ElasticsearchFilterOnlyRetriever,
    FilterRetriever,
    TfidfRetriever,
)
from haystack.nodes.retriever.text2sparql import Text2SparqlRetriever
from haystack.nodes.retriever.multimodal import MultiModalRetriever
