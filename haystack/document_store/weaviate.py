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
    
    Weaviate is a cloud-native, modular, real-time vector search engine built to scale your machine learning models.
    (See https://www.semi.technology/developers/weaviate/current/index.html#what-is-weaviate)
    
    Some of the key differences in contrast to FAISS & Milvus:
    1. Stores everything in one place: documents, meta data and vectors - so less network overhead when scaling this up
    2. Allows combination of vector search and scalar filtering, i.e. you can filter for a certain tag and do dense retrieval on that subset 
    3. Has less variety of ANN algorithms, as of now only HNSW.  

    Weaviate python client is used to connect to the server, more details are here
    https://weaviate-python-client.readthedocs.io/en/docs/weaviate.html

    Usage:
    1. Start a Weaviate server (see https://www.semi.technology/developers/weaviate/current/getting-started/installation.html)
    2. Init a WeaviateDocumentStore in Haystack
    """

    def __init__(
            self,
            host: Union[str, List[str]] = "http://localhost",
            port: Union[int, List[int]] = 8080,
            timeout_config: tuple = (5, 15),
            username: str = None,
            password: str = None,
            index: str = "Document",
            embedding_dim: int = 768,
            text_field: str = "text",
            name_field: str = "name",
            faq_question_field = "question",
            similarity: str = "dot_product",
            index_type: str = "hnsw",
            custom_schema: Optional[dict] = None,
            return_embedding: bool = False,
            embedding_field: str = "embedding",
            progress_bar: bool = True,
            duplicate_documents: str = 'overwrite',
            **kwargs,
    ):
        """
        :param host: Weaviate server connection URL for storing and processing documents and vectors.
                             For more details, refer "https://www.semi.technology/developers/weaviate/current/getting-started/installation.html"
        :param port: port of Weaviate instance
        :param timeout_config: Weaviate Timeout config as a tuple of (retries, time out seconds).
        :param username: username (standard authentication via http_auth)
        :param password: password (standard authentication via http_auth)
        :param index: Index name for document text, embedding and metadata (in Weaviate terminology, this is a "Class" in Weaviate schema).
        :param embedding_dim: The embedding vector size. Default: 768.
        :param text_field: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
                           If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
        :param name_field: Name of field that contains the title of the the doc
        :param faq_question_field: Name of field containing the question in case of FAQ-Style QA
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default.
        :param index_type: Index type of any vector object defined in weaviate schema. The vector index type is pluggable.
                           Currently, HSNW is only supported.
                           See: https://www.semi.technology/developers/weaviate/current/more-resources/performance.html
        :param custom_schema: Allows to create custom schema in Weaviate, for more details
                           See https://www.semi.technology/developers/weaviate/current/data-schema/schema-configuration.html
        :param module_name : Vectorization module to convert data into vectors. Default is "text2vec-trasnformers"
                            For more details, See https://www.semi.technology/developers/weaviate/current/modules/
        :param return_embedding: To return document embedding.
        :param embedding_field: Name of field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents:Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already exists.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            host=host, port=port, timeout_config=timeout_config, username=username, password=password,
            index=index, embedding_dim=embedding_dim, text_field=text_field, name_field=name_field,
            faq_question_field=faq_question_field, similarity=similarity, index_type=index_type,
            custom_schema=custom_schema,return_embedding=return_embedding, embedding_field=embedding_field,
            progress_bar=progress_bar, duplicate_documents=duplicate_documents
        )

        # Connect to Weaviate server using python binding
        weaviate_url =f"{host}:{port}"
        if username and password:
            secret = AuthClientPassword(username, password)
            self.weaviate_client = client.Client(url=weaviate_url,
                                             auth_client_secret=secret,
                                             timeout_config=timeout_config)
        else:
            self.weaviate_client = client.Client(url=weaviate_url,
                                                 timeout_config=timeout_config)

        # Test Weaviate connection
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
        self.index = index
        self.embedding_dim = embedding_dim
        self.text_field = text_field
        self.name_field = name_field
        self.faq_question_field = faq_question_field
        self.similarity = similarity
        self.index_type = index_type
        self.custom_schema = custom_schema
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        self._create_schema_and_index_if_not_exist(self.index)

    def _create_schema_and_index_if_not_exist(
        self,
        index: Optional[str] = None,
    ):
        """Create a new index (schema/class in Weaviate) for storing documents in case if an index (schema) with the name doesn't exist already."""
        index = index or self.index

        if self.custom_schema:
            schema = self.custom_schema
        else:
            schema = {
                    "classes": [
                        {
                            "class": index,
                            "description": "Haystack index, it's a class in Weaviate",
                            "invertedIndexConfig": {
                                "cleanupIntervalSeconds": 60
                            },
                            "vectorizer": "none",
                            "properties": [
                                {
                                    "dataType": [
                                        "string"
                                    ],
                                    "description": "Name Field",
                                    "name": self.name_field
                                },
                                {
                                    "dataType": [
                                        "string"
                                    ],
                                    "description": "Question Field",
                                    "name": self.faq_question_field
                                },
                                {
                                    "dataType": [
                                        "text"
                                    ],
                                    "description": "Document Text",
                                    "name": self.text_field
                                },
                            ],
                        }
                    ]
                }
        if not self.weaviate_client.schema.contains(schema):
            self.weaviate_client.schema.create(schema)

    def _convert_weaviate_result_to_document(
            self,
            result: dict,
            return_embedding: bool
    ) -> Document:
        """
        Convert weaviate result dict into haystack document object. This is more involved because
        weaviate search result dict varies between get and query interfaces.
        Weaviate get methods return the data items in properties key, whereas the query doesn't.
        """
        score = None
        probability = None
        text = ""
        question = None

        id = result.get("id")
        embedding = result.get("vector")

        # If properties key is present, get all the document fields from it.
        # otherwise, a direct lookup in result root dict
        props = result.get("properties")
        if not props:
            props = result

        if props.get(self.text_field) is not None:
            text = str(props.get(self.text_field))

        if props.get(self.faq_question_field) is not None:
            question = props.get(self.faq_question_field)

        # Weaviate creates "_additional" key for semantic search
        if "_additional" in props:
            if "certainty" in props["_additional"]:
                score = props["_additional"]['certainty']
                probability = score
            if "id" in props["_additional"]:
                id = props["_additional"]['id']
            if "vector" in props["_additional"]:
                embedding = props["_additional"]['vector']
            props.pop("_additional", None)

        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in props.items() if k not in (self.text_field, self.faq_question_field, self.embedding_field)}

        if return_embedding and embedding:
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
        # Sample result dict from a get method
        '''{'class': 'Document',
         'creationTimeUnix': 1621075584724,
         'id': '1bad51b7-bd77-485d-8871-21c50fab248f',
         'properties': {'meta': "{'key1':'value1'}",
          'name': 'name_5',
          'text': 'text_5'},
         'vector': []}'''
        index = index or self.index
        document = None
        result = self.weaviate_client.data_object.get_by_id(id, with_vector=True)
        if result:
            document = self._convert_weaviate_result_to_document(result, return_embedding=True)
        return document

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None,
                            batch_size: int = 10_000) -> List[Document]:
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        documents = []
        #TODO: better implementation with multiple where filters instead of chatty call below?
        for id in ids:
            result = self.weaviate_client.data_object.get_by_id(id, with_vector=True)
            if result:
                document = self._convert_weaviate_result_to_document(result, return_embedding=True)
                documents.append(document)
        return documents

    def _get_current_properties(self, index: Optional[str] = None) -> List[str]:
        """Get all the existing properties in the schema"""
        index = index or self.index
        cur_properties = []
        for class_item in self.weaviate_client.schema.get()['classes']:
            if class_item['class'] == index:
                cur_properties = [item['name'] for item in class_item['properties']]

        return cur_properties

    def _build_filter_clause(self, filters:Dict[str, List[str]]) -> dict:
        """Transform Haystack filter conditions to Weaviate where filter clauses"""
        weaviate_filters = []
        weaviate_filter = {}
        for key, values in filters.items():
            for value in values:
                weaviate_filter = {
                    "path": [key],
                    "operator": "Equal",
                    "valueString": value
                }
                weaviate_filters.append(weaviate_filter)
        if len(weaviate_filters) > 1:
            filter_dict = {
                "operator": "Or",
                "operands": weaviate_filters
            }
            return filter_dict
        else:
            return weaviate_filter

    def _update_schema(self, new_prop:str, index: Optional[str] = None):
        """Updates the schema with a new property"""
        index = index or self.index
        property_dict = {
            "dataType": [
                "string"
            ],
            "description": f"dynamic property {new_prop}",
            "name": new_prop
        }
        self.weaviate_client.schema.property.create(index, property_dict)

    def _check_document(self, cur_props: List[str], doc: dict) -> List[str]:
        """Find the properties in the document that don't exist in the existing schema"""
        return [item for item in doc.keys() if item not in cur_props]

    def write_documents(
            self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
            batch_size: int = 10_000, duplicate_documents: Optional[str] = None):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. Passing an Embedding/Vector is mandatory in case weaviate is not
                        configured with a module. If a module is configured, the embedding is automatically generated by Weaviate.
        :param index: index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        index = index or self.index
        self._create_schema_and_index_if_not_exist(index)
        field_map = self._create_document_field_map()

        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if len(documents) == 0:
            logger.warning("Calling DocumentStore.write_documents() with empty list")
            return

        # Auto schema feature https://github.com/semi-technologies/weaviate/issues/1539
        # Get and cache current properties in the schema
        current_properties = self._get_current_properties(index)

        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(document_objects, duplicate_documents)

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

                    # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
                    # we "unnest" all value within "meta"
                    if "meta" in _doc.keys():
                        for k, v in _doc["meta"].items():
                            _doc[k] = v
                        _doc.pop("meta")

                    doc_id = str(_doc.pop("id"))
                    vector = _doc.pop(self.embedding_field)
                    if _doc.get(self.faq_question_field) is None:
                        _doc.pop(self.faq_question_field)

                    # Check if additional properties are in the document, if so,
                    # append the schema with all the additional properties
                    missing_props = self._check_document(current_properties, _doc)
                    if missing_props:
                        for property in missing_props:
                            self._update_schema(property, index)
                            current_properties.append(property)

                    docs_batch.add(_doc, class_name=index, uuid=doc_id, vector=vector)

                # Ingest a batch of documents
                results = self.weaviate_client.batch.create(docs_batch)
                # Weaviate returns errors for every failed document in the batch
                if results is not None:
                    for result in results:
                        if 'result' in result and 'errors' in result['result'] \
                                and 'error' in result['result']['errors']:
                            for message in result['result']['errors']['error']:
                                logger.error(f"{message['message']}")
                progress_bar.update(batch_size)
        progress_bar.close()

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        self.weaviate_client.data_object.update(meta, class_name=self.index, uuid=id)

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index
        doc_count = 0
        if filters:
            filter_dict = self._build_filter_clause(filters=filters)
            result = self.weaviate_client.query.aggregate(index) \
                .with_fields("meta { count }") \
                .with_where(filter_dict)\
                .do()
        else:
            result = self.weaviate_client.query.aggregate(index)\
                    .with_fields("meta { count }")\
                    .do()

        if "data" in result:
            if "Aggregate" in result.get('data'):
                doc_count = result.get('data').get('Aggregate').get(index)[0]['meta']['count']

        return doc_count

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
        index = index or self.index
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def _get_all_documents_in_index(
        self,
        index: Optional[str],
        filters: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        index = index or self.index

        # Build the properties to retrieve from Weaviate
        properties = self._get_current_properties(index)
        properties.append("_additional {id, certainty, vector}")

        if filters:
            filter_dict = self._build_filter_clause(filters=filters)
            result = self.weaviate_client.query.get(class_name=index, properties=properties)\
                .with_where(filter_dict)\
                .do()
        else:
            result = self.weaviate_client.query.get(class_name=index, properties=properties)\
                .do()

        all_docs = {}
        if result and "data" in result and "Get" in result.get("data"):
            if result.get("data").get("Get").get(index):
                all_docs = result.get("data").get("Get").get(index)

        yield from all_docs

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
        query: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by Weaviate semantic search.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param custom_query: Custom query that will executed using query.raw method, for more details refer
                            https://www.semi.technology/developers/weaviate/current/graphql-references/filters.html
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        index = index or self.index

        # Build the properties to retrieve from Weaviate
        properties = self._get_current_properties(index)
        properties.append("_additional {id, certainty, vector}")

        if custom_query:
            query_output = self.weaviate_client.query.raw(custom_query)
        elif filters:
            filter_dict = self._build_filter_clause(filters)
            query_output = self.weaviate_client.query\
                .get(class_name=index, properties=properties)\
                .with_where(filter_dict)\
                .with_limit(top_k)\
                .do()
        else:
            raise NotImplementedError("Weaviate does not support inverted index text query. However, "
                                      "it allows to search by filters example : {'text': 'some text'} or "
                                      "use a custom GraphQL query in text format!")

        results = []
        if query_output and "data" in query_output and "Get" in query_output.get("data"):
            if query_output.get("data").get("Get").get(index):
                results = query_output.get("data").get("Get").get(index)

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
        :param index: index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """
        if return_embedding is None:
            return_embedding = self.return_embedding
        index = index or self.index

        # Build the properties to retrieve from Weaviate
        properties = self._get_current_properties(index)
        properties.append("_additional {id, certainty, vector}")

        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        query_string = {
            "vector" : query_emb
        }
        if filters:
            filter_dict = self._build_filter_clause(filters)
            query_output = self.weaviate_client.query\
                .get(class_name=index, properties=properties)\
                .with_where(filter_dict)\
                .with_near_vector(query_string)\
                .with_limit(top_k)\
                .do()
        else:
            query_output = self.weaviate_client.query\
                .get(class_name=index, properties=properties)\
                .with_near_vector(query_string)\
                .with_limit(top_k)\
                .do()

        results = []
        if query_output and "data" in query_output and "Get" in query_output.get("data"):
            if query_output.get("data").get("Get").get(index):
                results = query_output.get("data").get("Get").get(index)

        documents = []
        for result in results:
            doc = self._convert_weaviate_result_to_document(result, return_embedding=return_embedding)
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
        This can be useful if want to change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Weaviate mandates an embedding while creating the document itself.
        This option must be always true for weaviate and it will update the embeddings for all the documents.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing WeaviateDocumentStore()")

        if update_existing_embeddings:
            logger.info(f"Updating embeddings for all {self.get_document_count(index=index)} docs ...")
        else:
            raise RuntimeError("All the documents in Weaviate store have an embedding by default. Only update is allowed!")

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
        )

        for result_batch in get_batches_from_generator(result, batch_size):
            document_batch = [self._convert_weaviate_result_to_document(hit, return_embedding=False) for hit in result_batch]
            embeddings = retriever.embed_passages(document_batch)  # type: ignore
            assert len(document_batch) == len(embeddings)

            if embeddings[0].shape[0] != self.embedding_dim:
                raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                   f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                   "Specify the arg `embedding_dim` when initializing WeaviateDocumentStore()")
            for doc, emb in zip(document_batch, embeddings):
                # Using update method to only update the embeddings, other properties will be in tact
                self.weaviate_client.data_object.update({}, class_name=index, uuid=doc.id, vector=emb)

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        index = index or self.index
        if filters:
            docs_to_delete = self.get_all_documents(index, filters=filters)
            for doc in docs_to_delete:
                self.weaviate_client.data_object.delete(doc.id)
        else:
            self.weaviate_client.schema.delete_class(index)
            self._create_schema_and_index_if_not_exist(index)
