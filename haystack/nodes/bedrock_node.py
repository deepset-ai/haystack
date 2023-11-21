from haystack.nodes.base import BaseComponent
from haystack.schema import Document
import numpy as np
import boto3
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, Tuple


class BedrockEmbeddingRetriever(BaseComponent):
    outgoing_edges = 1
    
    """
    Custom pipeline node logic since Haystack doesnt currently support Bedrock as a retriever.

    :param docstore: The docstore to use (FAISS etc.).
    :param bedrock_client: The boto3(bedrock-runtime) client to be used for the invoke_model operation.
    :usage: retriever = BedrockEmbeddingRetriever(document_store, client)
    """

    def __init__(self, document_store, bedrock_client):
        self.client = bedrock_client
        self.document_store = document_store

    def _get_embeddings(self, text: str) -> List[float]:
        input_body = {}
        input_body["inputText"] = text
        body = json.dumps(input_body)
        response = self.client.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            return response_body.get("embedding")

    
    def get_document_embeddings(self, documents: List[str]):
        doc_embeds = []
        for doc in documents:
            response = self._get_embeddings(doc)
            doc_embeds.append(response)
        return doc_embeds

    def run(self, documents: List[Document]) -> tuple[dict[str, list[Document]], str]:
        content = [d.content for d in documents]
        metadata = [d.meta for d in documents]
        
        embeddings = self.get_document_embeddings(content)
        np_embedded = np.array(embeddings, dtype=np.float32)

        docs = []

        for i in range(0, len(embeddings)):
            docs.append(Document(content=content[i], meta=metadata[i], embedding=np_embedded[i]))

        output = {
            "documents": docs,
        }
        return output, "output_1"

    def run_batch(self, documents: List[Document]) -> tuple[dict[str, list[Document]], str]:
        pass    
    

class BedrockContextRetriever(BaseComponent):
    outgoing_edges = 1

    """
    Custom pipeline node logic since Haystack doesnt currently support Bedrock as a retriever.

    :param docstore: The docstore to use (FAISS etc.).
    :param top_k: Nearest neighbors to fetch.
    :param filters: Document_name to filter results by.
    :param bedrock_client: The boto3(bedrock-runtime) client to be used for the invoke_model operation.
    :usage: retriever = BedrockEmbeddingRetriever(document_store, top_k=5, filters="document.pdf", bedrock_client=client)
    """

    def __init__(self, document_store, top_k, filters, bedrock_client):
        self.client = bedrock_client
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k

    def _get_embeddings(self, text: str):
        input_body = {}
        input_body["inputText"] = text
        body = json.dumps(input_body)
        response = self.client.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            return response_body.get("embedding")

    def run(self, query) -> tuple[dict[str, list[Document]], str]:
        document_store = self.document_store
        filters = self.filters
        embedded_q = np.array([self._get_embeddings(query)], dtype=np.float32)

        if filters is not None:
            num_docs_to_check = 30
            res = document_store.query_by_embedding(embedded_q, top_k=num_docs_to_check)
        else:
            res = document_store.query_by_embedding(embedded_q, top_k=self.top_k)

        if filters is not None:
            cont = []
            for r in res:
                if r.meta['document_name'] == filters:
                    cont.append(r)
            if len(cont) == 0:
                cont = [Document(content="No Context Found", meta={"document_name": "N/A", "score": 0, "page": "N/A"})]
                return cont
            top_k_cont = cont[:self.top_k]
            answers = top_k_cont
        else:    
            answers = res

        output = {
            "answers": answers,
        }
        return output, "output_1"
