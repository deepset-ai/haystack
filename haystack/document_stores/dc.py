from typing import List, Optional, Union, Dict, Any, Generator

import json
import logging
import requests
import numpy as np

from haystack.document_stores import KeywordDocumentStore
from haystack.schema import Document, Label, MultiLabel

DEFAULT_API_ENDPOINT = f"DC_API/v1"

logger = logging.getLogger(__name__)


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class DCDocumentStore(KeywordDocumentStore):   
    def __init__(
        self, 
        api_key: str, 
        workspace: str = "default", 
        index: str = "default", 
        duplicate_documents: str = 'overwrite',
        api_endpoint: Optional[str] = None):
        """
        A DocumentStore facade enabling you to interact with the documents stored in DC.
        Thus you can run experiments like trying new nodes, pipelines, etc. without having to index your data again.
        
        DCDocumentStore is not intended to be used in production-like scenarios.

        :param api_key: Secret value of the API key (altenative authentication mode to the above http_auth)
        :param workspace: workspace in DC
        :param index: index to access within the DC workspace
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.

        """
        self.api_key = api_key
        self.workspace = workspace
        self.index = index
        self.label_index = index
        self.duplicate_documents = duplicate_documents

        if api_endpoint is None:
            api_endpoint = DEFAULT_API_ENDPOINT
        self.api_endpoint = api_endpoint

        init_url = self._get_index_endpoint()       
        response = requests.get(init_url, auth=BearerAuth(self.api_key))
        if response.status_code != 200:
            raise Exception(f"Could not connect to DC: HTTP {response.status_code} - {response.reason}")

        res = response.json()
        self.similarity = res["similarity"]
        self.return_embedding = res["return_embedding"]

        self.set_config(
            workspace=workspace, index=index, duplicate_documents=duplicate_documents,
            api_endpoint=self.api_endpoint
        )

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None, 
            batch_size: int = 10_000, 
            headers: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        logging.warning("`get_all_documents()` can get very slow and resource-heavy since all documents must be loaded from DC. "
                        "Consider using `get_all_documents_generator()` instead.")
        return list(self.get_all_documents_generator(index=index, filters=filters, 
                                return_embedding=return_embedding, batch_size=batch_size, headers=headers))

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000, 
        headers: Optional[Dict[str, str]] = None
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if batch_size != 10_000:
            raise ValueError("DCDocumentStore does not support batching")

        if return_embedding is not None and return_embedding != self.return_embedding:
            raise ValueError("DCDocumentStore does not support dynamic return_embeddings values")

        if index is None:
            index = self.index

        body: dict = {}
        if filters is not None:
            body["filters"] = filters
        if return_embedding is not None:
            body["return_embedding"] = return_embedding
        
        url = f"{self._get_index_endpoint(index)}/documents-stream"
        response = requests.post(url=url, json=body, stream=True, headers=headers, auth=BearerAuth(self.api_key))
        for raw_doc in response.iter_lines():
            dict_doc = json.loads(raw_doc.decode('utf-8'))
            yield Document.from_dict(dict_doc)

    def get_document_by_id(self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Document]:
        if index is None:
            index = self.index
        
        url = f"{self._get_index_endpoint(index)}/documents/{id}"
        response = requests.get(url=url, headers=headers, auth=BearerAuth(self.api_key))
        if response.status_code == 200:
            doc_dict = response.json()
            return Document.from_dict(doc_dict)
        else:
            logger.warning(f"Document {id} could not be fetched from DC: HTTP {response.status_code} - {response.reason}")
            return None

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None,
                            batch_size: int = 10_000, headers: Optional[Dict[str, str]] = None) -> List[Document]:
        if batch_size != 10_000:
            raise ValueError("DCDocumentStore does not support batching")
        
        docs = (self.get_document_by_id(id, index=index, headers=headers) for id in ids)
        return [doc for doc in docs if doc is not None]

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Optional[Dict[str, List[str]]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :param headers: Custom HTTP headers to pass to requests
        :return:
        """
        return self._query_documents(query_emb=query_emb, 
                                    filters=filters, 
                                    top_k=top_k, 
                                    index=index,
                                    return_embedding=return_embedding,
                                    headers=headers)

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to requests
        """
        return self._query_documents(query=query, 
                                    filters=filters, 
                                    top_k=top_k,
                                    custom_query=custom_query,
                                    index=index,
                                    headers=headers)

    def _query_documents(
        self,
        query: Optional[str] = None,
        query_emb: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None
    ) -> List[Document]:
        if index is None:
            index = self.index
        
        body = {
            "query": query,
            "query_emb": query_emb.tolist() if query_emb is not None else None,
            "filters": filters,
            "top_k": top_k,
            "return_embedding": return_embedding,
            "custom_query": custom_query
        }

        # remove null values from json
        body = {k:v for k,v in body.items() if v is not None}
        
        url = f"{self._get_index_endpoint(index)}/documents-query"
        response = requests.post(url=url, json=body, headers=headers, auth=BearerAuth(self.api_key))
        if response.status_code != 200:
            raise Exception(f"error during query: HTTP {response.status_code} - {response.reason}")
        
        doc_dicts = response.json()
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def _get_index_endpoint(self, index: Optional[str] = None) -> str:
        if index is None:
            index = self.index
        
        return f"{self.api_endpoint}/workspaces/{self.workspace}/indexes/{index}"
