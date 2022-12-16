from functools import wraps
from typing import List, Optional, Union, Dict, Generator, Any

import json
import logging
import numpy as np

from haystack.document_stores import KeywordDocumentStore
from haystack.errors import HaystackError
from haystack.schema import Document, FilterType, Label
from haystack.utils import DeepsetCloud, DeepsetCloudError, args_to_kwargs

logger = logging.getLogger(__name__)


def disable_and_log(func):
    """
    Decorator to disable write operation, shows warning and inputs instead.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.disabled_write_warning_shown:
            logger.warning(
                "Note that DeepsetCloudDocumentStore does not support write operations. "
                "In order to verify your pipeline works correctly, each input to write operations will be logged."
            )
            self.disabled_write_warning_shown = True

        args_as_kwargs = args_to_kwargs(args, func)
        parameters = {**args_as_kwargs, **kwargs}
        logger.info("Input to %s: %s", func.__name__, parameters)

    return wrapper


class DeepsetCloudDocumentStore(KeywordDocumentStore):
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: str = "default",
        index: Optional[str] = None,
        duplicate_documents: str = "overwrite",
        api_endpoint: Optional[str] = None,
        similarity: str = "dot_product",
        return_embedding: bool = False,
        label_index: str = "default",
        embedding_dim: int = 768,
    ):
        """
        A DocumentStore facade enabling you to interact with the documents stored in deepset Cloud.
        Thus you can run experiments like trying new nodes, pipelines, etc. without having to index your data again.

        You can also use this DocumentStore to create new pipelines on deepset Cloud. To do that, take the following
        steps:

        - create a new DeepsetCloudDocumentStore without an index (e.g. `DeepsetCloudDocumentStore()`)
        - create query and indexing pipelines using this DocumentStore
        - call `Pipeline.save_to_deepset_cloud()` passing the pipelines and a `pipeline_config_name`
        - call `Pipeline.deploy_on_deepset_cloud()` passing the `pipeline_config_name`

        DeepsetCloudDocumentStore is not intended for use in production-like scenarios.
        See [https://haystack.deepset.ai/components/document-store](https://haystack.deepset.ai/components/document-store)
        for more information.

        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
                        See docs on how to generate an API key for your workspace: https://docs.cloud.deepset.ai/docs/connect-deepset-cloud-to-your-application
        :param workspace: workspace name in deepset Cloud
        :param index: name of the index to access within the deepset Cloud workspace. This equals typically the name of
                      your pipeline. You can run Pipeline.list_pipelines_on_deepset_cloud() to see all available ones.
                      If you set index to `None`, this DocumentStore will always return empty results.
                      This is especially useful if you want to create a new Pipeline within deepset Cloud
                      (see Pipeline.save_to_deepset_cloud()` and `Pipeline.deploy_on_deepset_cloud()`).
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If DEEPSET_CLOUD_API_ENDPOINT environment variable is not specified either, defaults to "https://api.cloud.deepset.ai/api/v1".
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default since it is
                           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence Transformer model.
        :param label_index: index for the evaluation set interface
        :param return_embedding: To return document embedding.
        :param embedding_dim: Specifies the dimensionality of the embedding vector (only needed when using a dense retriever, for example, DensePassageRetriever pr EmbeddingRetriever, on top).
        """
        self.index = index
        self.label_index = label_index
        self.duplicate_documents = duplicate_documents
        self.similarity = similarity
        self.return_embedding = return_embedding
        self.embedding_dim = embedding_dim
        self.client = DeepsetCloud.get_index_client(
            api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, index=index
        )
        # Check if index exists
        pipeline_client = DeepsetCloud.get_pipeline_client(
            api_key=api_key, api_endpoint=api_endpoint, workspace=workspace
        )
        deployed_pipelines = set()
        deployed_unhealthy_pipelines = set()
        try:
            for pipe in pipeline_client.list_pipeline_configs(workspace=workspace):
                if pipe["status"] == "DEPLOYED":
                    deployed_pipelines.add(pipe["name"])
                elif pipe["status"] == "DEPLOYED_UNHEALTHY":
                    deployed_unhealthy_pipelines.add(pipe["name"])
        except Exception as ie:
            raise DeepsetCloudError(f"Could not connect to deepset Cloud:\n{ie}") from ie

        self.index_exists = index in deployed_pipelines | deployed_unhealthy_pipelines

        if self.index_exists:
            index_info = self.client.info()
            indexing_info = index_info["indexing"]
            if indexing_info["pending_file_count"] > 0:
                logger.warning(
                    f"{indexing_info['pending_file_count']} files are pending to be indexed. "
                    f"Indexing status: {indexing_info['status']}"
                )
            if index in deployed_unhealthy_pipelines:
                logger.warning(
                    f"The index '{index}' is unhealthy and should be redeployed using "
                    f"`Pipeline.undeploy_on_deepset_cloud()` and `Pipeline.deploy_on_deepset_cloud()`."
                )
        else:
            logger.info(
                f"You are using a DeepsetCloudDocumentStore with an index that does not exist on deepset Cloud. "
                f"This document store always returns empty responses. This can be useful if you want to "
                f"create a new pipeline within deepset Cloud.\n"
                f"In order to create a new pipeline on deepset Cloud, take the following steps: \n"
                f"  - create query and indexing pipelines using this DocumentStore\n"
                f"  - call `Pipeline.save_to_deepset_cloud()` passing the pipelines and a `pipeline_config_name`\n"
                f"  - call `Pipeline.deploy_on_deepset_cloud()` passing the `pipeline_config_name`"
            )

        self.evaluation_set_client = DeepsetCloud.get_evaluation_set_client(
            api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, evaluation_set=label_index
        )

        self.disabled_write_warning_shown = False

        super().__init__()

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
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
            "`get_all_documents()` can get very slow and resource-heavy since all documents must be loaded from deepset Cloud. "
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
        filters: Optional[FilterType] = None,
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
        if not self.index_exists:
            return

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
        if not self.index_exists:
            return None

        if index is None:
            index = self.index

        doc_dict = self.client.get_document(id=id, index=index, headers=headers)
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
        if not self.index_exists:
            return []

        if batch_size != 10_000:
            raise ValueError("DeepsetCloudDocumentStore does not support batching")

        docs = (self.get_document_by_id(id, index=index, headers=headers) for id in ids)
        return [doc for doc in docs if doc is not None]

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        if not self.index_exists:
            return 0

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
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
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
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :return:
        """
        if not self.index_exists:
            return []

        if return_embedding is None:
            return_embedding = self.return_embedding

        doc_dicts = self.client.query(
            query_emb=query_emb.tolist(),
            filters=filters,
            top_k=top_k,
            return_embedding=return_embedding,
            index=index,
            scale_score=scale_score,
            headers=headers,
        )
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def query(
        self,
        query: Optional[str],
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = True,
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
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if not self.index_exists:
            return []

        doc_dicts = self.client.query(
            query=query,
            filters=filters,
            top_k=top_k,
            custom_query=custom_query,
            index=index,
            all_terms_must_match=all_terms_must_match,
            scale_score=scale_score,
            headers=headers,
        )
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def query_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = True,
    ) -> List[List[Document]]:
        # TODO This method currently just calls query multiple times. Adapt this once there is a query_batch endpoint
        # in DC.

        documents = []
        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        for query, cur_filters in zip(queries, filters):
            cur_docs = self.query(
                query=query,
                filters=cur_filters,
                top_k=top_k,
                custom_query=custom_query,
                index=index,
                headers=headers,
                all_terms_must_match=all_terms_must_match,
                scale_score=scale_score,
            )
            documents.append(cur_docs)

        return documents

    def _create_document_field_map(self) -> Dict:
        return {}

    @disable_and_log
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
        pass

    @disable_and_log
    def update_document_meta(self, id: str, meta: Dict[str, Any], index: Optional[str] = None):
        """
        Update the metadata dictionary of a document by specifying its string id.

        :param id: The ID of the Document whose metadata is being updated.
        :param meta: A dictionary with key-value pairs that should be added / changed for the provided Document ID.
        :param index: Name of the index the Document is located at.
        """
        pass

    def get_evaluation_sets(self) -> List[dict]:
        """
        Returns a list of uploaded evaluation sets to deepset cloud.

        :return: list of evaluation sets as dicts
                 These contain ("name", "evaluation_set_id", "created_at", "matched_labels", "total_labels") as fields.
        """
        return self.evaluation_set_client.get_evaluation_sets()

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        """
        Returns a list of labels for the given index name.

        :param index: Optional name of evaluation set for which labels should be searched.
                      If None, the DocumentStore's default label_index (self.label_index) will be used.
        :filters: Not supported.
        :param headers: Not supported.

        :return: list of Labels.
        """
        return self.evaluation_set_client.get_labels(evaluation_set=index)

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Counts the number of labels for the given index and returns the value.

        :param index: Optional evaluation set name for which the labels should be counted.
                      If None, the DocumentStore's default label_index (self.label_index) will be used.
        :param headers: Not supported.

        :return: number of labels for the given index
        """
        return self.evaluation_set_client.get_labels_count(evaluation_set=index)

    @disable_and_log
    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        pass

    @disable_and_log
    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        pass

    @disable_and_log
    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        pass

    @disable_and_log
    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        pass

    @disable_and_log
    def delete_index(self, index: str):
        pass
