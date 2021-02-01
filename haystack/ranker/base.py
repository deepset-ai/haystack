import logging
from typing import Any, Optional, Dict, List, Union

from haystack import Document
from haystack.document_store.base import BaseDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.retriever.dense import EmbeddingRetriever
from haystack.document_store.memory import InMemoryDocumentStore

logger = logging.getLogger(__name__)

class BaseRanker:
    document_store: BaseDocumentStore
    retriever: BaseRetriever
    outgoing_edges = 1

    def __init__(self,
                 embedding_model: str,
                 use_gpu: bool = True,
                 model_format: str = "farm",
                 pooling_strategy: str = "reduce_mean",
                 emb_extraction_layer: int = -1,
                 top_k_ranker: int = None):

        self.document_store = InMemoryDocumentStore(index="document",
                                                    label_index="label",
                                                    embedding_field="embedding",
                                                    embedding_dim=512,
                                                    similarity="cosine")

        self.retriever = EmbeddingRetriever(document_store=self.document_store,
                                            embedding_model=embedding_model,
                                            use_gpu=use_gpu, model_format=model_format,
                                            pooling_strategy=pooling_strategy,
                                            emb_extraction_layer=emb_extraction_layer)

        self.top_k_ranker = top_k_ranker

    def run(self, query: str, documents: List[Document], top_k_ranker: Optional[int] = None, **kwargs):
        if documents:
            self.document_store.write_documents(documents)
            self.document_store.update_embeddings(retriever=self.retriever)

            if top_k_ranker:
                documents = self.retriever.retrieve(query=query, top_k=top_k_ranker)
            elif self.top_k_ranker:
                documents = self.retriever.retrieve(query=query, top_k=self.top_k_ranker)
            else:
                documents = self.retriever.retrieve(query=query)

            self.document_store.delete_all_documents(index='document')

            document_ids = [doc.id for doc in documents]
            logger.debug(f"Retrieved documents with IDs: {document_ids}")
            output = {
                "query": query,
                "documents": documents,
                **kwargs
            }
        else:
            logger.debug(f"No documents were passed")
            output = {
                "query": query,
                "documents": documents,
                **kwargs
            }

        return output, "output_1"