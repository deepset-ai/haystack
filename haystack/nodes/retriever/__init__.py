from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.dense import (
    DensePassageRetriever,
    DenseRetriever,
    EmbeddingRetriever,
    MultihopEmbeddingRetriever,
    TableTextRetriever,
)
from haystack.nodes.retriever.multimodal import MultiModalRetriever
from haystack.nodes.retriever.sparse import (
    BM25Retriever,
    ElasticsearchFilterOnlyRetriever,
    ElasticsearchRetriever,
    FilterRetriever,
    TfidfRetriever,
)
from haystack.nodes.retriever.text2sparql import Text2SparqlRetriever
