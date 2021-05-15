import logging
from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np
from tqdm import tqdm

from haystack import Document
from haystack.document_store.base import BaseDocumentStore
from haystack.utils import get_batches_from_generator

from weaviate import client, auth, AuthClientPassword
from weaviate import ObjectsBatchRequest

logger = logging.getLogger(__name__)

class WeaviateDocumentStore(BaseDocumentStore):
    """
    Milvus (https://milvus.io/) is a highly reliable, scalable Document Store specialized on storing and processing vectors.
    Therefore, it is particularly suited for Haystack users that work with dense retrieval methods (like DPR).
    In contrast to FAISS, Milvus ...
     - runs as a separate service (e.g. a Docker container) and can scale easily in a distributed environment
     - allows dynamic data management (i.e. you can insert/delete vectors without recreating the whole index)
     - encapsulates multiple ANN libraries (FAISS, ANNOY ...)

    This class uses Milvus for all vector related storage, processing and querying.
    The meta-data (e.g. for filtering) and the document text are however stored in a separate SQL Database as Milvus
    does not allow these data types (yet).

    Usage:
    1. Start a Weaviate server (see https://www.semi.technology/developers/weaviate/current/getting-started/installation.html)
    2. Init a WeaviateDocumentStore in Haystack
    """

    def __init__(
            self,
            weaviate_url: str = "http://localhost:8080",
            timeout_config: tuple = (5, 15),
            username: str = None,
            password: str = None,
            index: str = "Document",
            vector_dim: int = 768,
            text_field: str = "text",
            name_field: str = "name",
            faq_question_field = "question",
            similarity: str = "dot_product",
            index_type: str = "hnsw",
            custom_schema: Optional[dict] = None,
            module_name: str = "text2vec-transformers",
            index_param: Optional[Dict[str, Any]] = None,
            search_param: Optional[Dict[str, Any]] = None,
            update_existing_documents: bool = False,
            return_embedding: bool = False,
            embedding_field: str = "embedding",
            progress_bar: bool = True,
            **kwargs,
    ):
        """
        :param sql_url: SQL connection URL for storing document texts and metadata. It defaults to a local, file based SQLite DB. For large scale
                        deployment, Postgres is recommended. If using MySQL then same server can also be used for
                        Milvus metadata. For more details see https://milvus.io/docs/v1.0.0/data_manage.md.
        :param milvus_url: Milvus server connection URL for storing and processing vectors.
                           Protocol, host and port will automatically be inferred from the URL.
                           See https://milvus.io/docs/v1.0.0/install_milvus.md for instructions to start a Milvus instance.
        :param connection_pool: Connection pool type to connect with Milvus server. Default: "SingletonThread".
        :param index: Index name for text, embedding and metadata (in Milvus terms, this is the "collection name").
        :param vector_dim: The embedding vector size. Default: 768.
        :param index_file_size: Specifies the size of each segment file that is stored by Milvus and its default value is 1024 MB.
         When the size of newly inserted vectors reaches the specified volume, Milvus packs these vectors into a new segment.
         Milvus creates one index file for each segment. When conducting a vector search, Milvus searches all index files one by one.
         As a rule of thumb, we would see a 30% ~ 50% increase in the search performance after changing the value of index_file_size from 1024 to 2048.
         Note that an overly large index_file_size value may cause failure to load a segment into the memory or graphics memory.
         (From https://milvus.io/docs/v1.0.0/performance_faq.md#How-can-I-get-the-best-performance-from-Milvus-through-setting-index_file_size)
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
                           'cosine' is recommended for Sentence Transformers, but is not directly supported by Milvus.
                           However, you can normalize your embeddings and use `dot_product` to get the same results.
                           See https://milvus.io/docs/v1.0.0/metric.md?Inner-product-(IP)#floating.
        :param index_type: Type of approximate nearest neighbour (ANN) index used. The choice here determines your tradeoff between speed and accuracy.
                           Some popular options:
                           - FLAT (default): Exact method, slow
                           - IVF_FLAT, inverted file based heuristic, fast
                           - HSNW: Graph based, fast
                           - ANNOY: Tree based, fast
                           See: https://milvus.io/docs/v1.0.0/index.md
        :param index_param: Configuration parameters for the chose index_type needed at indexing time.
                            For example: {"nlist": 16384} as the number of cluster units to create for index_type IVF_FLAT.
                            See https://milvus.io/docs/v1.0.0/index.md
        :param search_param: Configuration parameters for the chose index_type needed at query time
                             For example: {"nprobe": 10} as the number of cluster units to query for index_type IVF_FLAT.
                             See https://milvus.io/docs/v1.0.0/index.md
        :param update_existing_documents: Whether to update any existing documents with the same ID when adding
                                          documents. When set as True, any document with an existing ID gets updated.
                                          If set to False, an error is raised if the document ID of the document being
                                          added already exists.
        :param return_embedding: To return document embedding.
        :param embedding_field: Name of field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            weaviate_url=weaviate_url, timeout_config=timeout_config, username=username, password=password,
            index=index, index_type=index_type, custom_schema=custom_schema, module_name=module_name,
            vector_dim=vector_dim, text_field=text_field, name_field=name_field, faq_question_field=faq_question_field,
            similarity=similarity, index_param=index_param,search_param=search_param, update_existing_documents=update_existing_documents,
            return_embedding=return_embedding, embedding_field=embedding_field, progress_bar=progress_bar,
        )

        if username and password:
            secret = AuthClientPassword(username, password)
            self.weaviate_client = client.Client(url=weaviate_url,
                                             auth_client_secret=secret,
                                             timeout_config=timeout_config)
        else:
            self.weaviate_client = client.Client(url=weaviate_url,
                                                 timeout_config=timeout_config)

        # Test connection
        try:
            status = self.weaviate_client.is_ready()
            if not status:
                raise ConnectionError(
                    f"Initial connection to Weaviate failed. Make sure you run Weaviate instance "
                    f"at `{weaviate_url}` and that it has finished the initial ramp up (can take > 30s)."
                )
        except Exception:
            raise ConnectionError(
                f"Initial connection to Weaviate failed. Make sure you run Weaviate instance "
                f"at `{weaviate_url}` and that it has finished the initial ramp up (can take > 30s)."
            )

        self.vector_dim = vector_dim
        self.text_field = text_field
        self.name_field = name_field
        self.faq_question_field = faq_question_field
        self.index_type = index_type
        self.custom_schema = custom_schema
        self.module_name = module_name
        self.index = index
        self.index_param = index_param or {"nlist": 16384}
        self.search_param = search_param or {"nprobe": 10}
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar

        self._create_schema_and_index_if_not_exist(self.index)

    #def __del__(self):
        #return self.milvus_server.close()

    def _create_schema_and_index_if_not_exist(
        self,
        index: Optional[str] = None,
        index_param: Optional[Dict[str, Any]] = None
    ):
        index = index or self.index
        index_param = index_param or self.index_param

        if self.custom_schema:
            schema = self.custom_schema
        else:
            schema = {
                    "classes": [
                        {
                            "class": index,
                            "description": "Haystack default class",
                            "invertedIndexConfig": {
                                "cleanupIntervalSeconds": 60
                            },
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "poolingStrategy": "masked_mean",
                                    "vectorizeClassName": False
                                }
                            },
                            "properties": [
                                {
                                    "dataType": [
                                        "string"
                                    ],
                                    "description": "Name Field",
                                    "moduleConfig": {
                                        "text2vec-transformers": {
                                            "skip": False,
                                            "vectorizePropertyName": False
                                        }
                                    },
                                    "name": self.name_field
                                },
                                {
                                    "dataType": [
                                        "string"
                                    ],
                                    "description": "Question Field",
                                    "moduleConfig": {
                                        "text2vec-transformers": {
                                            "skip": False,
                                            "vectorizePropertyName": False
                                        }
                                    },
                                    "name": self.faq_question_field
                                },
                                {
                                    "dataType": [
                                        "text"
                                    ],
                                    "description": "Document Text",
                                    "moduleConfig": {
                                        "text2vec-transformers": {
                                            "skip": True,
                                            "vectorizePropertyName": True
                                        }
                                    },
                                    "name": self.text_field
                                },
                                {
                                    "dataType": [
                                        "string"
                                    ],
                                    "description": "Document meta",
                                    "moduleConfig": {
                                        "text2vec-transformers": {
                                            "skip": False,
                                            "vectorizePropertyName": False
                                        }
                                    },
                                    "name": "meta"
                                }
                            ],
                            "vectorIndexConfig": {
                                "cleanupIntervalSeconds": 300,
                                "maxConnections": 64,
                                "efConstruction": 128,
                                "vectorCacheMaxObjects": 500000
                            },
                            "vectorIndexType": "hnsw",
                            "vectorizer": "text2vec-transformers"
                        }
                    ]
                }
        if not self.weaviate_client.schema.contains(schema):
            self.weaviate_client.schema.create(schema)
        else:
            logger.warning(f"Schema already exists in Weaviate {schema}")

    def _convert_weaviate_result_to_document(
            self,
            result: dict,
            return_embedding: bool
    ) -> Document:
        import ast
        # By default, the result json will have the following fields
        id = result.get("id")
        embedding = result.get("vector")
        score = None
        probability = None

        # Weaviate Get method returns the data items in properties key,
        # Weaviate query doesn't have a properties key.
        props = result.get("properties")
        if not props:
            props = result

        text = props.get(self.text_field)
        question = props.get(self.faq_question_field)

        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in props.items() if k not in (self.text_field, self.faq_question_field, self.embedding_field)}
        name = meta_data.pop(self.name_field, None)
        if name:
            meta_data["name"] = name

        if result.get("_additional"):
            score = result.get("_additional").get('certainty') if result.get("_additional").get('certainty') else None
            if score:
                probability = score

            id = result.get("_additional").get('id') if result.get("_additional").get('id') else None

        if return_embedding:
            if not embedding:
                embedding = result.get("_additional").get("vector")
            if embedding:
                embedding = np.asarray(embedding, dtype=np.float32)

        document = Document(
            id=id,
            text=text,
            meta=meta_data,
            score=score,
            probability=probability,
            question=question,
            embedding=embedding,
        )
        return document

    def _create_document_field_map(self) -> Dict:
        return {
            self.text_field: "text",
            self.embedding_field: "embedding",
            self.faq_question_field if self.faq_question_field else "question": "question"
        }

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        '''{'class': 'Document',
         'creationTimeUnix': 1621075584724,
         'id': '1bad51b7-bd77-485d-8871-21c50fab248f',
         'properties': {'meta': "{'key1':'value1'}",
          'name': 'name_5',
          'text': 'text_5'},
         'vector': []}'''
        index = index or self.index
        result = self.weaviate_client.data_object.get_by_id(id, with_vector=True)
        document = self._convert_weaviate_result_to_document(result, return_embedding=True)
        return document

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        docs = []
        for id in ids:
            docs.append(self.get_document_by_id(id))
        return docs

    def write_documents(
            self, documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000
    ):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                                  them right away in Milvus. If not, you can later call update_embeddings() to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return:
        """
        index = index or self.index
        self._create_schema_and_index_if_not_exist(index)
        field_map = self._create_document_field_map()

        if len(documents) == 0:
            logger.warning("Calling DocumentStore.write_documents() with empty list")
            return

        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(total=len(document_objects), disable=not self.progress_bar) as progress_bar:
            for document_batch in batched_documents:
                docs_batch = ObjectsBatchRequest()
                for idx, doc in enumerate(document_batch):
                    _doc = {
                        **doc.to_dict(field_map=self._create_document_field_map())
                    }
                    _ = _doc.pop("score", None)
                    _ = _doc.pop("probability", None)
                    if "meta" in _doc.keys():
                        _doc["meta"] = str(_doc.get("meta"))
                    doc_id = str(_doc.pop("id"))
                    vector = _doc.pop(self.embedding_field)
                    if _doc.get(self.faq_question_field) is None:
                        _doc.pop(self.faq_question_field)
                    if vector:
                        docs_batch.add(_doc, class_name=self.index, uuid=doc_id, vector=vector)
                    else:
                        docs_batch.add(_doc, class_name=self.index, uuid=doc_id)

                outputs = self.weaviate_client.batch.create(docs_batch)
                for output in outputs:
                    if output.get('result').get('errors'):
                        print(output.get('result').get('errors'))
                progress_bar.update(batch_size)
        progress_bar.close()

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        body = {"meta": meta}
        self.weaviate_client.data_object.update(body, class_name=self.index, uuid=id)

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the number of documents in the document store.
        """
        pass

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

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        where_filter = {
            "operator": "And",
            "operands": []
        }

        if filters:
            operands = []
            for key, values in filters.items():
                operands.append(
                    {
                        "path": [key],
                        "operator": "equal",
                        "valueString": [values]
                    }
                )
                where_filter = {
                    "operator": "And",
                    "operands": [operands]
                }

        if only_documents_without_embedding:
            raise OSError()

        result = self.weaviate_client.query.get(class_name=self.index, properties=[self.text_field,"_additional {id, certainty}"])\
            .with_limit(batch_size)\
            .do()
        yield from result.get("data").get("Get").get(self.index)

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

        results = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size)
        for result in results:
            document = self._convert_weaviate_result_to_document(result, return_embedding=return_embedding)
            yield document

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

        if filters:
            logger.warning("Query filters are not implemented for the WeaviateDocumentStore.")

        index = index or self.index

        if custom_query:
            query_output = self.weaviate_client.query.raw(custom_query)
        else:
            query_string = {
                "concepts" : [query]
            }
            query_output = self.weaviate_client.query\
                .get(class_name=index, properties=[self.text_field,"_additional {id, certainty}"])\
                .with_near_text(query_string)\
                .with_limit(top_k)\
                .do()

        results = query_output.get("data").get("Get").get(self.index)
        documents = []
        for result in results:
            doc = self._convert_weaviate_result_to_document(result, return_embedding=True)
            documents.append(doc)

        return documents

    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: (SQL) index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """
        if filters:
            logger.warning("Query filters are not implemented for the WeaviateDocumentStore.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        index = index or self.index

        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        query_string = {
            "vector" : query_emb
        }
        query_output = self.weaviate_client.query\
            .get(class_name=index, properties=[self.text_field,"_additional {id, certainty}"])\
            .with_near_vector(query_string)\
            .with_limit(top_k)\
            .do()

        results = query_output.get("data").get("Get").get(self.index)
        documents = []
        for result in results:
            doc = self._convert_weaviate_result_to_document(result, return_embedding=True)
            documents.append(doc)

        return documents

    def update_embeddings(
        self,
        retriever,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        update_existing_embeddings: bool = True,
        batch_size: int = 10_000
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        raise RuntimeError("Weaviate store produces embeddings by default based on the configuration in "
                           "schema. Update embeddings isn't implemented for this store!")


    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete all documents (from SQL AND Milvus).
        :param index: (SQL) index name for storing the docs and metadata
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :return: None
        """
        index = index or self.index
        self.weaviate_client.schema.delete_class(index)

