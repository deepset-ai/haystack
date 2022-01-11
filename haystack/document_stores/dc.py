from typing import List, Optional, Union, Dict, Any, Generator

import json
import logging
import requests

from haystack.document_stores import BaseDocumentStore
from haystack.schema import Document, Label, MultiLabel

DC_BASE_URL = "https://dccloud"
DC_API = f"{DC_BASE_URL}/v1"

class DCDocumentStore(BaseDocumentStore):   
    def __init__(
        self, 
        api_key: str, 
        workspace: str = "default", 
        index: str = "default", 
        duplicate_documents: str = 'overwrite'):
        self.api_key = api_key
        self.workspace = workspace
        self.index = index
        self.label_index = index
        self.duplicate_documents = duplicate_documents

        init_url = self._get_index_endpoint()       
        res = requests.get(init_url).json()
        self.similarity = res.similarity
        self.return_embedding = res.return_embedding

        self.set_config(
            api_key=api_key, workspace=workspace, index=index, duplicate_documents=duplicate_documents,
            similarity=self.similarity
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

        response = requests.get(f"{self._get_index_endpoint(index)}/documents", stream=True, headers=headers)
        for raw_doc in response.iter_lines():
            dict_doc = json.loads(raw_doc.decode('utf-8'))
            yield Document.from_dict(dict_doc)

    def _get_index_endpoint(self, index: Optional[str] = None) -> str:
        if index is None:
            index = self.index
        
        return f"{DC_API}/workspaces/{self.workspace}/indexes/{index}"
