from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.dense import (
    DensePassageRetriever,
    DenseRetriever,
    EmbeddingRetriever,
    MultihopEmbeddingRetriever,
    TableTextRetriever,
)
from haystack.nodes.retriever.multimodal import MultiModalRetriever
from haystack.nodes.retriever.sparse import BM25Retriever, FilterRetriever, TfidfRetriever
from haystack.nodes.retriever.web import WebRetriever
