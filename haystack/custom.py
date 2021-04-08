from typing import List
from haystack.schema import BaseComponent
from haystack.retriever.dense import EmbeddingRetriever
import numpy as np
from haystack import Document


class TitleEmbeddingRetriever(EmbeddingRetriever):
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings of the titles for a list of passages. For this Retriever type: The same as calling .embed()
        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.meta['name'] for d in docs]

        return self.embed(texts)


class JoinDocumentsCustom(BaseComponent):
    """
    A node to join documents outputted by multiple retriever nodes.
    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    """

    outgoing_edges = 1

    def __init__(
        self, ks_retriever: List[int] = None
    ):
        """
        :param ks_retriever: A node-wise list(length of list must be equal to the number of input nodes) of k_retriever kept for
                        the concatenation of the retrievers in the nodes. If set to None, the number of documents retrieved will be used
        """
        self.ks_retriever = ks_retriever

    def run(self, **kwargs):
        inputs = kwargs["inputs"]

        document_map = {}
        if self.ks_retriever:
            ks_retriever = self.ks_retriever
        else:
            ks_retriever = [len(inputs[0]['documents']) for i in range(len(inputs))]
        for input_from_node, k_retriever in zip(inputs, ks_retriever):
            for i, doc in enumerate(input_from_node["documents"]):
                if i == k_retriever:
                    break
                document_map[doc.id] = doc
        documents = document_map.values()
        output = {"query": inputs[0]["query"], "documents": documents, "labels": inputs[0].get("labels", None)}
        return output, "output_1"
