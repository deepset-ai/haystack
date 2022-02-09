from typing import List, Optional, Union, Dict, Generator

import json
import logging
import numpy as np

from haystack.document_stores import KeywordDocumentStore
from haystack.schema import Document, Label
from haystack.utils import DeepsetCloud


DEFAULT_API_ENDPOINT = f"DC_API_PLACEHOLDER/v1"  # TODO

logger = logging.getLogger(__name__)


class DeepsetCloudDocumentStore(KeywordDocumentStore):
    def __init__(
        self,
        api_key: str = None,
        workspace: str = "default",
        index: str = "default",
        duplicate_documents: str = "overwrite",
        api_endpoint: Optional[str] = None,
        similarity: str = "dot_product",
        return_embedding: bool = False,
    ):
        """
        A DocumentStore facade enabling you to interact with the documents stored in Deepset Cloud.
        Thus you can run experiments like trying new nodes, pipelines, etc. without having to index your data again.

        DeepsetCloudDocumentStore is not intended for use in production-like scenarios.
        See https://haystack.deepset.ai/components/document-store for more information.

        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param workspace: workspace in Deepset Cloud
        :param index: index to access within the Deepset Cloud workspace
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default since it is
                           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
        :param return_embedding: To return document embedding.

        """
        self.index = index
        self.label_index = index
        self.duplicate_documents = duplicate_documents
        self.similarity = similarity
        self.return_embedding = return_embedding
        self.client = DeepsetCloud.get_index_client(
            api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, index=index
        )

        index_info = self.client.info()
        indexing_info = index_info["indexing"]
        if indexing_info["pending_file_count"] > 0:
            logger.warning(
                f"{indexing_info['pending_file_count']} files are pending to be indexed. Indexing status: {indexing_info['status']}"
            )

        self.set_config(
            workspace=workspace,
            index=index,
            duplicate_documents=duplicate_documents,
            api_endpoint=api_endpoint,
            similarity=similarity,
            return_embedding=return_embedding,
        )

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        logging.warning(
            "`get_all_documents()` can get very slow and resource-heavy since all documents must be loaded from Deepset Cloud. "
            "Consider using `get_all_documents_generator()` instead."
        )
        return list(
            self.get_all_documents_generator(
                index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size, headers=headers
            )
        )

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if batch_size != 10_000:
            raise ValueError("DeepsetCloudDocumentStore does not support batching")

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        raw_documents = self.client.stream_documents(
            return_embedding=return_embedding, filters=filters, index=index, headers=headers
        )
        for raw_doc in raw_documents:
            dict_doc = json.loads(raw_doc.decode("utf-8"))
            yield Document.from_dict(dict_doc)

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        if index is None:
            index = self.index

        doc_dict = self.client.get_document(id=id, return_embedding=self.return_embedding, index=index, headers=headers)
        doc: Optional[Document] = None
        if doc_dict:
            doc = Document.from_dict(doc_dict)

        return doc

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if batch_size != 10_000:
            raise ValueError("DeepsetCloudDocumentStore does not support batching")

        docs = (self.get_document_by_id(id, index=index, headers=headers) for id in ids)
        return [doc for doc in docs if doc is not None]

    def get_document_count(
        self,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        count_result = self.client.count_documents(
            filters=filters,
            only_documents_without_embedding=only_documents_without_embedding,
            index=index,
            headers=headers,
        )
        return count_result["count"]

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :param headers: Custom HTTP headers to pass to requests
        :return:
        """
        if return_embedding is None:
            return_embedding = self.return_embedding

        doc_dicts = self.client.query(
            query_emb=query_emb.tolist(),
            filters=filters,
            similarity=self.similarity,
            top_k=top_k,
            return_embedding=return_embedding,
            index=index,
            headers=headers,
        )
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param custom_query: Custom query to be executed.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to requests
        """
        doc_dicts = self.client.query(
            query=query, filters=filters, top_k=top_k, custom_query=custom_query, index=index, headers=headers
        )
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def _create_document_field_map(self) -> Dict:
        return {}

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

        :return: None
        """
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support writing documents.")

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support labels.")

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support labels.")

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support labels.")

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support deleting documents.")

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support deleting documents.")

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("DeepsetCloudDocumentStore currently does not support labels.")
