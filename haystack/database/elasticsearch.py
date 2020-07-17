import json
import logging
from string import Template
from typing import List, Optional, Union, Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

from haystack.database.base import BaseDocumentStore, Document

logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: str = "",
        password: str = "",
        index: str = "document",
        search_fields: Union[str, list] = "text",
        text_field: str = "text",
        name_field: str = "name",
        external_source_id_field: str = "external_source_id",
        embedding_field: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        custom_mapping: Optional[dict] = None,
        excluded_meta_data: Optional[list] = None,
        faq_question_field: Optional[str] = None,
        scheme: str = "http",
        ca_certs: bool = False,
        verify_certs: bool = True,
        create_index: bool = True
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
        :param external_source_id_field: If you have an external id (= non-elasticsearch) that identifies your documents, you can specify it here.
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param embedding_dim: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param custom_mapping: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
        :param excluded_meta_data: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
                                   Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
        :param scheme: 'https' or 'http', protocol used to connect to your elasticsearch instance
        :param ca_certs: Root certificates for SSL
        :param verify_certs: Whether to be strict about ca certificates
        :param create_index: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case)
        """
        self.client = Elasticsearch(hosts=[{"host": host, "port": port}], http_auth=(username, password),
                                    scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs)

        # if no custom_mapping is supplied, use the default mapping
        if not custom_mapping:
            custom_mapping = {
                "mappings": {
                    "properties": {
                        name_field: {"type": "text"},
                        text_field: {"type": "text"},
                        external_source_id_field: {"type": "text"},
                    }
                }
            }
            if embedding_field:
                custom_mapping["mappings"]["properties"][embedding_field] = {"type": "dense_vector",
                                                                             "dims": embedding_dim}
        # create an index if not exists
        if create_index:
            self.client.indices.create(index=index, ignore=400, body=custom_mapping)
        self.index = index

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        #TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
        # custom fields (e.g. meta data you want to return)
        self.search_fields = search_fields
        self.text_field = text_field
        self.name_field = name_field
        self.external_source_id_field = external_source_id_field
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.excluded_meta_data = excluded_meta_data
        self.faq_question_field = faq_question_field

    def get_document_by_id(self, id: str) -> Optional[Document]:
        query = {"query": {"ids": {"values": [id]}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]

        document = self._convert_es_hit_to_document(result[0]) if result else None
        return document

    def get_document_ids_by_tags(self, tags: dict) -> List[str]:
        term_queries = [{"terms": {key: value}} for key, value in tags.items()]
        query = {"query": {"bool": {"must": term_queries}}}
        logger.debug(f"Tag filter query: {query}")
        result = self.client.search(index=self.index, body=query, size=10000)["hits"]["hits"]
        doc_ids = []
        for hit in result:
            doc_ids.append(hit["_id"])
        return doc_ids

    def write_documents(self, documents: List[dict]):
        """
        Indexes documents for later queries in Elasticsearch.

        :param documents: List of dictionaries.
                          Default format: {"text": "<the-actual-text>"}
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
                          Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
                          should be changed to what you have set for self.text_field and self.name_field .
        :return: None
        """

        documents_to_index = []
        for doc in documents:
            _doc = {
                "_op_type": "create",
                "_index": self.index,
                **doc
            }  # type: Dict[str, Any]

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)
        bulk(self.client, documents_to_index, request_timeout=300)

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        body = {"doc": meta}
        self.client.update(index=self.index, doc_type="_doc", id=id, body=body)

    def get_document_count(self, index: Optional[str] = None,) -> int:
        if index is None:
            index = self.index
        result = self.client.count(index=index)
        count = result["count"]
        return count

    def get_all_documents(self, index=None) -> List[Document]:
        if index is None:
            index = self.index
        result = scan(self.client, query={"query": {"match_all": {}}}, index=index)
        documents = [self._convert_es_hit_to_document(hit) for hit in result]
        return documents

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
    ) -> List[Document]:

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
        elif custom_query:  # substitute placeholder for question and filters for the custom_query template string
            template = Template(custom_query)
            # replace all "${question}" placeholder(s) with query
            substitutions = {"question": query}
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

        documents = [self._convert_es_hit_to_document(hit) for hit in result]
        return documents

    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None) -> List[Document]:
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in cosine similarity to avoid negative numbers
            body= {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector,doc['{self.embedding_field}']) + 1.0",
                            "params": {
                                "query_vector": query_emb
                            }
                        }
                    }
                }
            }  # type: Dict[str,Any]

            if filters:
                filter_clause = []
                for key, values in filters.items():
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

            documents = [self._convert_es_hit_to_document(hit, score_adjustment=-1) for hit in result]
            return documents

    def _convert_es_hit_to_document(self, hit: dict, score_adjustment: int = 0) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in hit["_source"].items() if k not in (self.text_field, self.external_source_id_field)}
        meta_data["name"] = meta_data.pop(self.name_field, None)

        document = Document(
            id=hit["_id"],
            text=hit["_source"][self.text_field],
            external_source_id=hit["_source"].get(self.external_source_id_field),
            meta=meta_data,
            query_score=hit["_score"] + score_adjustment if hit["_score"] else None,
            question=hit["_source"].get(self.faq_question_field)
        )
        return document

    def update_embeddings(self, retriever, index=None):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever
        :return: None
        """
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")



        docs = self.get_all_documents(index)
        passages = [d.text for d in docs]
        logger.info(f"Updating embeddings for {len(passages)} docs ...")
        embeddings = retriever.embed_passages(passages)

        assert len(docs) == len(embeddings)

        if embeddings[0].shape[0] != self.embedding_dim:
            raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                               f" doesn't match embedding dim. in documentstore ({self.embedding_dim})."
                               "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()")
        doc_updates = []
        for doc, emb in zip(docs, embeddings):
            update = {"_op_type": "update",
                      "_index": index,
                      "_id": doc.id,
                      "doc": {self.embedding_field: emb.tolist()},
                      }
            doc_updates.append(update)

        bulk(self.client, doc_updates, request_timeout=300)

    def add_eval_data(self, filename: str, doc_index: str = "eval_document", label_index: str = "feedback"):
        """
        Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.

        :param filename: Name of the file containing evaluation data
        :type filename: str
        :param doc_index: Elasticsearch index where evaluation documents should be stored
        :type doc_index: str
        :param label_index: Elasticsearch index where labeled questions should be stored
        :type label_index: str
        """

        eval_docs_to_index = []
        questions_to_index = []

        # Create index with eval docs if not existing
        default_mapping = {
            "mappings": {
                "properties": {
                    self.name_field: {"type": "text"},
                    self.text_field: {"type": "text"},
                    self.external_source_id_field: {"type": "text"},
                    self.embedding_field: {"type": "dense_vector", "dims": self.embedding_dim}
                }
            }
        }
        self.client.indices.create(index=doc_index, ignore=400, body=default_mapping)

        with open(filename, "r") as file:
            data = json.load(file)
            for document in data["data"]:
                for paragraph in document["paragraphs"]:
                    doc_to_index= {}
                    id = hash(paragraph["context"])
                    for fieldname, value in paragraph.items():
                        # write docs to doc_index
                        if fieldname == "context":
                            doc_to_index[self.text_field] = value
                            doc_to_index["doc_id"] = str(id)
                            doc_to_index["_op_type"] = "create"
                            doc_to_index["_index"] = doc_index
                        # write questions to label_index
                        elif fieldname == "qas":
                            for qa in value:
                                question_to_index = {
                                    "question": qa["question"],
                                    "answers": qa["answers"],
                                    "doc_id": str(id),
                                    "origin": "gold_label",
                                    "index_name": doc_index,
                                    "_op_type": "create",
                                    "_index": label_index
                                }
                                questions_to_index.append(question_to_index)
                        # additional fields for docs
                        else:
                            doc_to_index[fieldname] = value

                    for key, value in document.items():
                        if key == "title":
                            doc_to_index[self.name_field] = value
                        elif key != "paragraphs":
                            doc_to_index[key] = value

                    eval_docs_to_index.append(doc_to_index)

        bulk(self.client, eval_docs_to_index)
        bulk(self.client, questions_to_index)

    def get_all_documents_in_index(self, index: str, filters: Optional[dict] = None):
        body = {
            "query": {
                "bool": {
                    "must": {
                        "match_all" : {}
                    }
                }
            }
        }  # type: Dict[str, Any]

        if filters:
           body["query"]["bool"]["filter"] = {"term": filters}
        result = scan(self.client, query=body, index=index)

        return result
