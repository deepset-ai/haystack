import json
import logging
import time
from copy import deepcopy
from string import Template
from typing import List, Optional, Union, Dict, Any, Generator
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from elasticsearch.exceptions import RequestError
import numpy as np
from scipy.special import expit

from haystack.document_store.base import BaseDocumentStore
from haystack import Document, Label
from haystack.retriever.base import BaseRetriever
from haystack.utils import get_batches_from_generator

logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: str = "",
        password: str = "",
        index: str = "document",
        label_index: str = "label",
        search_fields: Union[str, list] = "text",
        text_field: str = "text",
        name_field: str = "name",
        embedding_field: str = "embedding",
        embedding_dim: int = 768,
        custom_mapping: Optional[dict] = None,
        excluded_meta_data: Optional[list] = None,
        faq_question_field: Optional[str] = None,
        analyzer: str = "standard",
        scheme: str = "http",
        ca_certs: str = None,
        verify_certs: bool = True,
        create_index: bool = True,
        update_existing_documents: bool = False,
        refresh_type: str = "wait_for",
        similarity="dot_product",
        timeout=30,
        return_embedding: bool = False,
    ):
        """
        A DocumentStore using Elasticsearch to store and query the documents for our search.

            * Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
            * You can either use an existing Elasticsearch index or create a new one via haystack
            * Retrievers operate on top of this DocumentStore to find the relevant documents for a query

        :param host: url of elasticsearch
        :param port: port of elasticsearch
        :param username: username
        :param password: password
        :param index: Name of index in elasticsearch to use. If not existing yet, we will create one.
        :param search_fields: Name of fields used by ElasticsearchRetriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
        :param text_field: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
                           If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
        :param name_field: Name of field that contains the title of the the doc
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param embedding_dim: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param custom_mapping: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
        :param analyzer: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
                         Elasticsearch also has built-in analyzers for different languages (e.g. impacting tokenization). More info at:
                         https://www.elastic.co/guide/en/elasticsearch/reference/7.9/analysis-analyzers.html
        :param excluded_meta_data: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
                                   Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
        :param scheme: 'https' or 'http', protocol used to connect to your elasticsearch instance
        :param ca_certs: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk. You can use certifi package with certifi.where() to find where the CA certs file is located in your machine.
        :param verify_certs: Whether to be strict about ca certificates
        :param create_index: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case)
        :param update_existing_documents: Whether to update any existing documents with the same ID when adding
                                          documents. When set as True, any document with an existing ID gets updated.
                                          If set to False, an error is raised if the document ID of the document being
                                          added already exists.
        :param refresh_type: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
                             If set to 'wait_for', continue only after changes are visible (slow, but safe).
                             If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
                             More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
                           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
        :param timeout: Number of seconds after which an ElasticSearch request times out.
        :param return_embedding: To return document embedding


        """
        self.client = Elasticsearch(hosts=[{"host": host, "port": port}], http_auth=(username, password),
                                    scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs, timeout=timeout)

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        #TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
        # custom fields (e.g. meta data you want to return)
        self.search_fields = search_fields
        self.text_field = text_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.excluded_meta_data = excluded_meta_data
        self.faq_question_field = faq_question_field
        self.analyzer = analyzer
        self.return_embedding = return_embedding

        self.custom_mapping = custom_mapping
        self.index: str = index
        self.label_index: str = label_index
        if similarity in ["cosine", "dot_product"]:
            self.similarity = similarity
        else:
            raise Exception("Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine' and 'dot_product'")
        if create_index:
            self._create_document_index(index)
            self._create_label_index(label_index)

        self.update_existing_documents = update_existing_documents
        self.refresh_type = refresh_type

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents. In case if an index with the name already exists, it ensures that
        the embedding_field is present.
        """
        # check if the existing index has the embedding field; if not create it
        if self.client.indices.exists(index=index_name):
            if self.embedding_field:
                mapping = self.client.indices.get(index_name)[index_name]["mappings"]
                if self.embedding_field in mapping["properties"] and mapping["properties"][self.embedding_field]["type"] != "dense_vector":
                    raise Exception(f"The '{index_name}' index in Elasticsearch already has a field called '{self.embedding_field}'"
                                    f" with the type '{mapping['properties'][self.embedding_field]['type']}'. Please update the "
                                    f"document_store to use a different name for the embedding_field parameter.")
                mapping["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}
                self.client.indices.put_mapping(index=index_name, body=mapping)
            return

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        self.name_field: {"type": "keyword"},
                        self.text_field: {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                }
            }
            if self.embedding_field:
                mapping["mappings"]["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}

        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _create_label_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "question": {"type": "text"},
                    "answer": {"type": "text"},
                    "is_correct_answer": {"type": "boolean"},
                    "is_correct_document": {"type": "boolean"},
                    "origin": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "offset_start_in_doc": {"type": "long"},
                    "no_answer": {"type": "boolean"},
                    "model_id": {"type": "keyword"},
                    "type": {"type": "keyword"},
                }
            }
        }
        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    # TODO: Add flexibility to define other non-meta and meta fields expected by the Document class
    def _create_document_field_map(self) -> Dict:
        return {
            self.text_field: "text",
            self.embedding_field: "embedding",
            self.faq_question_field if self.faq_question_field else "question": "question"
        }

    def get_document_by_id(self, id: str, index=None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        index = index or self.index
        documents = self.get_documents_by_id([id], index=index)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(self, ids: List[str], index=None) -> List[Document]:
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        query = {"query": {"ids": {"values": ids}}}
        result = self.client.search(index=index, body=query)["hits"]["hits"]
        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def write_documents(
        self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,  batch_size: int = 10_000
    ):
        """
        Indexes documents for later queries in Elasticsearch.

        Behaviour if a document with the same ID already exists in ElasticSearch:
        a) (Default) Throw Elastic's standard error message for duplicate IDs.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they will automatically get UUIDs assigned. See the `Document` class for details)

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
                          Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
                          should be changed to what you have set for self.text_field and self.name_field.
        :param index: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.
        :param batch_size: Number of documents that are passed to Elasticsearch's bulk function at a time.
        :return: None
        """

        if index and not self.client.indices.exists(index=index):
            self._create_document_index(index)

        if index is None:
            index = self.index

        documents_to_index = []
        for document in documents:
            # Make sure we comply to Document class format
            if isinstance(document, dict):
                doc = Document.from_dict(document, field_map=self._create_document_field_map())
            else:
                doc = document

            _doc = {
                "_op_type": "index" if self.update_existing_documents else "create",
                "_index": index,
                **doc.to_dict(field_map=self._create_document_field_map())
            }  # type: Dict[str, Any]

            # cast embedding type as ES cannot deal with np.array
            if _doc[self.embedding_field] is not None:
                if type(_doc[self.embedding_field]) == np.ndarray:
                    _doc[self.embedding_field] = _doc[self.embedding_field].tolist()

            # rename id for elastic
            _doc["_id"] = str(_doc.pop("id"))

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _ = _doc.pop("probability", None)
            _doc = {k:v for k,v in _doc.items() if v is not None}

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)
                documents_to_index = []

        if documents_to_index:
            bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)

    def write_labels(
        self, labels: Union[List[Label], List[dict]], index: Optional[str] = None, batch_size: int = 10_000
    ):
        """Write annotation labels into document store.

        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param batch_size: Number of labels that are passed to Elasticsearch's bulk function at a time.
        """
        index = index or self.label_index
        if index and not self.client.indices.exists(index=index):
            self._create_label_index(index)

        labels_to_index = []
        for l in labels:
            # Make sure we comply to Label class format
            if isinstance(l, dict):
                label = Label.from_dict(l)
            else:
                label = l

            _label = {
                "_op_type": "index" if self.update_existing_documents else "create",
                "_index": index,
                **label.to_dict()
            }  # type: Dict[str, Any]

            # rename id for elastic
            if label.id is not None:
                _label["_id"] = str(_label.pop("id"))

            labels_to_index.append(_label)

            # Pass batch_size number of labels to bulk
            if len(labels_to_index) % batch_size == 0:
                bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type)
                labels_to_index = []

        if labels_to_index:
            bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type)

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        body = {"doc": meta}
        self.client.update(index=self.index, id=id, body=body, refresh=self.refresh_type)

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index

        body: dict = {"query": {"bool": {}}}
        if filters:
            filter_clause = []
            for key, values in filters.items():
                if type(values) != list:
                    raise ValueError(
                        f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                        'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        result = self.client.count(index=index, body=body)
        count = result["count"]
        return count

    def get_label_count(self, index: Optional[str] = None) -> int:
        """
        Return the number of labels in the document store
        """
        return self.get_document_count(index=index)

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
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
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        result = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size)
        for hit in result:
            document = self._convert_es_hit_to_document(hit, return_embedding=return_embedding)
            yield document

    def get_all_labels(
        self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, batch_size: int = 10_000
    ) -> List[Label]:
        """
        Return all labels in the document store
        """
        index = index or self.label_index
        result = list(self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size))
        labels = [Label.from_dict(hit["_source"]) for hit in result]
        return labels

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 10_000,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        body = {
            "query": {
                "bool": {
                    "must": {
                        "match_all": {}
                    }
                }
            }
        }  # type: Dict[str, Any]

        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        result = scan(self.client, query=body, index=index, size=batch_size, scroll="1d")
        yield from result

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """

        if index is None:
            index = self.index

        # Naive retrieval without BM25, only filtering
        if query is None:
            body = {"query":
                        {"bool": {"must":
                                      {"match_all": {}}}}}  # type: Dict[str, Any]
            if filters:
                filter_clause = []
                for key, values in filters.items():
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

        # Retrieval via custom query
        elif custom_query:  # substitute placeholder for query and filters for the custom_query template string
            template = Template(custom_query)
            # replace all "${query}" placeholder(s) with query
            substitutions = {"query": query}
            # For each filter we got passed, we'll try to find & replace the corresponding placeholder in the template
            # Example: filters={"years":[2018]} => replaces {$years} in custom_query with '[2018]'
            if filters:
                for key, values in filters.items():
                    values_str = json.dumps(values)
                    substitutions[key] = values_str
            custom_query_json = template.substitute(**substitutions)
            body = json.loads(custom_query_json)
            # add top_k
            body["size"] = str(top_k)

        # Default Retrieval via BM25 using the user query on `self.search_fields`
        else:
            body = {
                "size": str(top_k),
                "query": {
                    "bool": {
                        "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                    }
                },
            }

            if filters:
                filter_clause = []
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                                         'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=index, body=body)["hits"]["hits"]

        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def query_by_embedding(self,
                           query_emb: np.array,
                           filters: Optional[Dict[str, List[str]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in similarity to avoid negative numbers (for cosine sim)
            body = {
                "size": top_k,
                "query": self._get_vector_similarity_query(query_emb, top_k)
            }
            if filters:
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                                         'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                body["query"]["script_score"]["query"] = {
                    "terms": filters
                }

            excluded_meta_data: Optional[list] = None

            if self.excluded_meta_data:
                excluded_meta_data = deepcopy(self.excluded_meta_data)

                if return_embedding is True and self.embedding_field in excluded_meta_data:
                    excluded_meta_data.remove(self.embedding_field)
                elif return_embedding is False and self.embedding_field not in excluded_meta_data:
                    excluded_meta_data.append(self.embedding_field)
            elif return_embedding is False:
                excluded_meta_data = [self.embedding_field]

            if excluded_meta_data:
                body["_source"] = {"excludes": excluded_meta_data}

            logger.debug(f"Retriever query: {body}")
            result = self.client.search(index=index, body=body, request_timeout=300)["hits"]["hits"]

            documents = [
                self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, return_embedding=return_embedding)
                for hit in result
            ]
            return documents

    def _get_vector_similarity_query(self, query_emb: np.array, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        if self.similarity == "cosine":
            similarity_fn_name = "cosineSimilarity"
        elif self.similarity == "dot_product":
            similarity_fn_name = "dotProduct"
        else:
            raise Exception("Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between \'cosine\' and \'dot_product\'")

        query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    # offset score to ensure a positive range as required by Elasticsearch
                    "source": f"{similarity_fn_name}(params.query_vector,'{self.embedding_field}') + 1000",
                    "params": {"query_vector": query_emb.tolist()},
                },
            }
        }
        return query

    def _convert_es_hit_to_document(
            self,
            hit: dict,
            return_embedding: bool,
            adapt_score_for_embedding: bool = False,

    ) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in hit["_source"].items() if k not in (self.text_field, self.faq_question_field, self.embedding_field)}
        name = meta_data.pop(self.name_field, None)
        if name:
            meta_data["name"] = name

        score = hit["_score"] if hit["_score"] else None
        if score:
            if adapt_score_for_embedding:
                score = self._scale_embedding_score(score)
                if self.similarity == "cosine":
                    probability = (score + 1) / 2  # scaling probability from cosine similarity
                elif self.similarity == "dot_product":
                    probability = float(expit(np.asarray(score / 100)))  # scaling probability from dot product
            else:
                probability = float(expit(np.asarray(score / 8)))  # scaling probability from TFIDF/BM25
        else:
            probability = None

        embedding = None
        if return_embedding:
            embedding_list = hit["_source"].get(self.embedding_field)
            if embedding_list:
                embedding = np.asarray(embedding_list, dtype=np.float32)

        document = Document(
            id=hit["_id"],
            text=hit["_source"].get(self.text_field),
            meta=meta_data,
            score=score,
            probability=probability,
            question=hit["_source"].get(self.faq_question_field),
            embedding=embedding,
        )
        return document

    def _scale_embedding_score(self, score):
        return score - 1000

    def describe_documents(self, index=None):
        """
        Return a summary of the documents in the document store
        """
        if index is None:
            index = self.index
        docs = self.get_all_documents(index)

        l = [len(d.text) for d in docs]
        stats = {"count": len(docs),
                 "chars_mean": np.mean(l),
                 "chars_max": max(l),
                 "chars_min": min(l),
                 "chars_median": np.median(l),
                 }
        return stats

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None, batch_size: int = 10_000):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")

        logger.info(f"Updating embeddings for {self.get_document_count(index=index)} docs ...")

        result = self.get_all_documents_generator(index, batch_size=batch_size)
        for document_batch in get_batches_from_generator(result, batch_size):
            if len(document_batch) == 0:
                break
            embeddings = retriever.embed_passages(document_batch)  # type: ignore
            assert len(document_batch) == len(embeddings)

            if embeddings[0].shape[0] != self.embedding_dim:
                raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                   f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                   "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()")
            doc_updates = []
            for doc, emb in zip(document_batch, embeddings):
                update = {"_op_type": "update",
                          "_index": index,
                          "_id": doc.id,
                          "doc": {self.embedding_field: emb.tolist()},
                          }
                doc_updates.append(update)

            bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type)

    def delete_all_documents(self, index: str, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        query: Dict[str, Any] = {"query": {}}
        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
                query["query"]["bool"] = {"filter": filter_clause}
        else:
            query["query"] = {"match_all": {}}
        self.client.delete_by_query(index=index, body=query, ignore=[404])
        # We want to be sure that all docs are deleted before continuing (delete_by_query doesn't support wait_for)
        time.sleep(1)


class OpenDistroElasticsearchDocumentStore(ElasticsearchDocumentStore):
    """
    Document Store using the Open Distro for Elasticsearch. It is compatible with the AWS Elasticsearch Service.

    In addition to native Elasticsearch query & filtering, it provides efficient vector similarity search using
    the KNN plugin that can scale to a large number of documents.
    """

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents.
        """

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        self.name_field: {"type": "keyword"},
                        self.text_field: {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                }
            }
            if self.embedding_field:
                if self.similarity == "cosine":
                    similarity_space_type = "cosinesimil"
                elif self.similarity == "dot_product":
                    similarity_space_type = "l2"
                else:
                    raise Exception(
                        f"Similarity function {self.similarity} is not supported by OpenDistroElasticsearchDocumentStore."
                    )
                mapping["settings"]["knn"] = True
                mapping["settings"]["knn.space_type"] = similarity_space_type
                mapping["mappings"]["properties"][self.embedding_field] = {
                    "type": "knn_vector",
                    "dimension": self.embedding_dim,
                }

        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _get_vector_similarity_query(self, query_emb: np.array, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        query = {"knn": {self.embedding_field: {"vector": query_emb.tolist(), "k": top_k}}}
        return query

    def _scale_embedding_score(self, score):
        return score