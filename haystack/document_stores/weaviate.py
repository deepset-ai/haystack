import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
from tqdm import tqdm

from haystack.schema import Document, FilterType, Label
from haystack.document_stores import KeywordDocumentStore
from haystack.utils.batching import get_batches_from_generator
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.document_stores.utils import convert_date_to_rfc3339
from haystack.errors import DocumentStoreError, HaystackError
from haystack.nodes.retriever import DenseRetriever
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install farm-haystack[weaviate]'") as weaviate_import:
    import weaviate
    from weaviate import AuthApiKey, AuthBearerToken, AuthClientCredentials, AuthClientPassword, client, gql

logger = logging.getLogger(__name__)
UUID_PATTERN = re.compile(r"^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$", re.IGNORECASE)
SUPPORTED_PROPERTY_TYPES = {
    "string",
    "string[]",
    "int",
    "int[]",
    "boolean",
    "boolean[]",
    "number",
    "number[]",
    "date",
    "date[]",
    "text",
    "text[]",
}


class WeaviateDocumentStoreError(DocumentStoreError):
    pass


class WeaviateDocumentStore(KeywordDocumentStore):
    """

    Weaviate is a cloud-native, modular, real-time vector search engine built to scale your machine learning models.
    (See https://weaviate.io/developers/weaviate/current/index.html#what-is-weaviate)

    Some of the key differences in contrast to FAISS:
    1. Stores everything in one place: documents, meta data and vectors - so less network overhead when scaling this up
    2. Allows combination of vector search and scalar filtering, i.e. you can filter for a certain tag and do dense retrieval on that subset
    3. Has less variety of ANN algorithms, as of now only HNSW.
    4. Requires document ids to be in uuid-format. If wrongly formatted ids are provided at indexing time they will be replaced with uuids automatically.

    Weaviate python client is used to connect to the server, more details are here
    https://weaviate.io/developers/weaviate/client-libraries/python

    Usage:
    1. Start a Weaviate server (see https://weaviate.io/developers/weaviate/current/getting-started/installation.html)
    2. Init a WeaviateDocumentStore in Haystack

    Connection Parameters Precedence:
    The selection and priority of connection parameters are as follows:
    1. If `use_embedded` is set to True, an embedded Weaviate instance will be used, and all other connection parameters will be ignored.
    2. If `use_embedded` is False or not provided and an `api_key` is provided, the `api_key` will be used to authenticate through AuthApiKey, assuming a connection to a Weaviate Cloud Service (WCS) instance.
    3. If neither `use_embedded` nor `api_key` is provided, but a `username` and `password` are provided, they will be used to authenticate through AuthClientPassword, assuming an OIDC Resource Owner Password flow.
    4. If none of the above conditions are met, no authentication method will be used and a connection will be attempted with the provided `host` and `port` values without any authentication.

    Limitations:
    The current implementation is not supporting the storage of labels, so you cannot run any evaluation workflows.
    """

    def __init__(
        self,
        host: Union[str, List[str]] = "http://localhost",
        port: Union[int, List[int]] = 8080,
        timeout_config: tuple = (5, 15),
        username: Optional[str] = None,
        password: Optional[str] = None,
        scope: Optional[str] = "offline_access",
        api_key: Optional[str] = None,
        use_embedded: bool = False,
        embedded_options: Optional[dict] = None,
        additional_headers: Optional[Dict[str, Any]] = None,
        index: str = "Document",
        embedding_dim: int = 768,
        content_field: str = "content",
        name_field: str = "name",
        similarity: str = "cosine",
        index_type: str = "hnsw",
        custom_schema: Optional[dict] = None,
        return_embedding: bool = False,
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        replication_factor: int = 1,
        batch_size: int = 10_000,
    ):
        """
        :param host: Weaviate server connection URL for storing and processing documents and vectors.
                             For more details, see [Weaviate installation](https://weaviate.io/developers/weaviate/current/getting-started/installation.html).
        :param port: The port of the Weaviate instance.
        :param timeout_config: The Weaviate timeout config as a tuple of (retries, time out seconds).
        :param username: The Weaviate username (standard authentication using http_auth).
        :param password: Weaviate password (standard authentication using http_auth).
        :param scope: The scope of the credentials when using the OIDC Resource Owner Password or Client Credentials authentication flow.
        :param api_key: The Weaviate Cloud Services (WCS) API key (for WCS authentication).
        :param use_embedded: Whether to use an embedded Weaviate instance. Default: False.
        :param embedded_options: Custom options for the embedded Weaviate instance. Default: None.
        :param additional_headers: Additional headers to be included in the requests sent to Weaviate, for example the bearer token.
        :param index: Index name for document text, embedding, and metadata (in Weaviate terminology, this is a "Class" in the Weaviate schema).
        :param embedding_dim: The embedding vector size. Default: 768.
        :param content_field: Name of the field that might contain the answer and is passed to the Reader model (for example, "full_text").
                           If no Reader is used (for example, in FAQ-Style QA), the plain content of this field is returned.
        :param name_field: Name of the field that contains the title of the doc.
        :param similarity: The similarity function used to compare document vectors. Available options are 'cosine' (default), 'dot_product', and 'l2'.
                           'cosine' is recommended for Sentence Transformers.
        :param index_type: Index type of any vector object defined in the Weaviate schema. The vector index type is pluggable.
                           Currently, only HSNW is supported.
                           See also [Weaviate documentation](https://weaviate.io/developers/weaviate/current/more-resources/performance.html).
        :param custom_schema: Allows to create a custom schema in Weaviate. For more details,
                           see [Weaviate documentation](https://weaviate.io/developers/weaviate/current/schema/schema-configuration.html).
        :param module_name: Vectorization module to convert data into vectors. Default is "text2vec-trasnformers"
                            For more details, see [Weaviate documentation](https://weaviate.io/developers/weaviate/current/modules/).
        :param return_embedding: Returns document embedding.
        :param embedding_field: Name of the field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents:Handle duplicates document based on parameter options.
                                    Parameter options: 'skip','overwrite','fail'
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: Raises an error if the document ID of the document being added already exists.
        :param recreate_index: If set to True, deletes an existing Weaviate index and creates a new one using the config you are using for initialization. Note that all data in the old index is
            lost if you choose to recreate the index.
        :param replication_factor: Sets the Weaviate Class's replication factor in Weaviate at the time of Class creation.
                                   See also [Weaviate documentation](https://weaviate.io/developers/weaviate/current/configuration/replication.html).
        :param batch_size: The number of documents to index at once.
        """
        weaviate_import.check()
        super().__init__()

        # Connect to Weaviate server using python binding
        if not use_embedded:
            weaviate_url = f"{host}:{port}"
            auth_client_secret = self._get_auth_secret(username, password, api_key, scope)
            embedded_options = None
        else:
            weaviate_url = None
            auth_client_secret = None
            embedded_options = self._get_embedded_options(embedded_options)

        # Timeout config can only be defined as a list in YAML, but Weaviate expects a tuple
        if isinstance(timeout_config, list):
            timeout_config = tuple(timeout_config)
        self.weaviate_client = client.Client(
            url=weaviate_url,
            embedded_options=embedded_options,
            auth_client_secret=auth_client_secret,
            timeout_config=timeout_config,
            additional_headers=additional_headers,
        )
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
        self.index = self._sanitize_index_name(index)
        self.embedding_dim = embedding_dim
        self.content_field = content_field
        self.name_field = name_field
        if similarity == "cosine":
            self.similarity = "cosine"
        elif similarity == "dot_product":
            self.similarity = "dot"
        elif similarity == "l2":
            self.similarity = "l2-squared"
        else:
            raise DocumentStoreError(
                f"It looks like you provided value '{similarity}' for similarity in the WeaviateDocumentStore constructor. Choose one of these values: 'cosine', 'l2', and 'dot_product'"
            )
        self.index_type = index_type
        self.custom_schema = custom_schema
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents
        self.replication_factor = replication_factor
        self.batch_size = batch_size

        self._create_schema_and_index(self.index, recreate_index=recreate_index)
        self.uuid_format_warning_raised = False

    @staticmethod
    def _get_auth_secret(
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        scope: Optional[str] = "offline_access",
    ) -> Optional[Union["AuthClientPassword", "AuthClientCredentials", "AuthBearerToken"]]:
        if api_key:
            return AuthApiKey(api_key=api_key)
        elif username and password:
            return AuthClientPassword(username, password, scope=scope)
        return None

    @staticmethod
    def _get_embedded_options(embedded_options: Optional[Dict[str, Any]] = None) -> "weaviate.EmbeddedOptions":
        embedded_options = embedded_options or {}
        return weaviate.EmbeddedOptions(**embedded_options)

    def _sanitize_index_name(self, index: Optional[str]) -> Optional[str]:
        if index is None:
            return None
        elif "_" in index:
            return "".join(x.capitalize() for x in index.split("_"))
        else:
            return index[0].upper() + index[1:]

    def _create_schema_and_index(self, index: Optional[str] = None, recreate_index: bool = False):
        """
        Create a new index (schema/class in Weaviate) for storing documents in case if an
        index (schema) with the name doesn't exist already.
        """
        index = self._sanitize_index_name(index) or self.index

        if self.custom_schema:
            schema = self.custom_schema
        else:
            schema = {
                "classes": [
                    {
                        "class": index,
                        "description": "Haystack index, it's a class in Weaviate",
                        "invertedIndexConfig": {"cleanupIntervalSeconds": 60},
                        "vectorizer": "none",
                        "properties": [
                            {"dataType": ["string"], "description": "Name Field", "name": self.name_field},
                            {
                                "dataType": ["text"],
                                "description": "Document Content (e.g. the text)",
                                "name": self.content_field,
                            },
                        ],
                        "vectorIndexConfig": {"distance": self.similarity},
                        "replicationConfig": {"factor": self.replication_factor},
                    }
                ]
            }

        if not self.weaviate_client.schema.contains(schema):
            self.weaviate_client.schema.create(schema)
        elif recreate_index and index is not None:
            self._delete_index(index)
            self.weaviate_client.schema.create(schema)
        else:
            # The index already exists in Weaviate. We need to check if the index's similarity metrics matches
            # the one this class is initialized with, as Weaviate doesn't allow switching similarity
            # metrics once an index alreadty exists in Weaviate.
            _db_similarity = self.weaviate_client.schema.get(class_name=index)["vectorIndexConfig"]["distance"]
            if _db_similarity != self.similarity:
                raise ValueError(
                    f"This index already exists in Weaviate with similarity '{_db_similarity}'. "
                    f"If there is a Weaviate index created with a certain similarity, you can't "
                    f"query with a different similarity. If you need a different similarity, "
                    f"recreate the index. To do this, set the `recreate_index=True` argument."
                )

    def _convert_weaviate_result_to_document(
        self,
        result: dict,
        return_embedding: bool,
        scale_score: bool = True,
        json_properties: Optional[List[str]] = None,
    ) -> Document:
        """
        Convert weaviate result dict into haystack document object. This is more involved because
        weaviate search result dict varies between get and query interfaces.
        Weaviate get methods return the data items in properties key, whereas the query doesn't.
        """
        json_properties = json_properties or []

        score = None
        content = ""

        # Sample result dict from a get method:
        # {
        #     'class': 'Document',
        #     'creationTimeUnix': 1621075584724,
        #     'id': '1bad51b7-bd77-485d-8871-21c50fab248f',
        #     'properties': {
        #         'meta': "{'key1':'value1'}",
        #         'name': 'name_5',
        #         'content': 'text_5'
        #     },
        #     'vector': []
        # }
        id = result.get("id")
        embedding = result.get("vector")

        # If properties key is present, get all the document fields from it.
        # otherwise, a direct lookup in result root dict
        props = result.get("properties")
        if not props:
            props = result

        if props.get(self.content_field) is not None:
            # Converting JSON-string to original datatype (string or nested list)
            content = json.loads(str(props.get(self.content_field)))

        content_type = None
        if props.get("content_type") is not None:
            content_type = str(props.pop("content_type"))

        id_hash_keys = None
        if props.get("id_hash_keys") is not None:
            id_hash_keys = props.pop("id_hash_keys")

        # Weaviate creates "_additional" key for semantic search
        if "_additional" in props:
            if "certainty" in props["_additional"]:
                score = props["_additional"]["certainty"]
                # weaviate returns already scaled values
                if score and not scale_score:
                    score = score * 2 - 1
            elif "distance" in props["_additional"]:
                score = props["_additional"]["distance"]
                if score:
                    # Weaviate returns the negative dot product. To make score comparable
                    # to other document stores, we take the negative.
                    if self.similarity == "dot":
                        score = -1 * score
                    if scale_score:
                        score = self.scale_to_unit_interval(score, self.similarity)
            if "id" in props["_additional"]:
                id = props["_additional"]["id"]
            if "vector" in props["_additional"]:
                embedding = props["_additional"]["vector"]
            props.pop("_additional", None)

        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {}
        for k, v in props.items():
            if k in (self.content_field, self.embedding_field):
                continue
            if v is None:
                continue
            # Weaviate doesn't support dict as a property type. We store them as JSON strings and convert them back here.
            if k in json_properties:
                if isinstance(v, list):
                    v = [json.loads(str(item)) for item in v]
                else:
                    v = json.loads(str(v))
            meta_data[k] = v

        document = Document.from_dict(
            {
                "id": id,
                "content": content,
                "content_type": content_type,
                "meta": meta_data,
                "score": score,
                "id_hash_keys": id_hash_keys,
            }
        )

        if return_embedding and embedding:
            document.embedding = np.asarray(embedding, dtype=np.float32)

        return document

    def _create_document_field_map(self) -> Dict:
        return {self.content_field: "content", self.embedding_field: "embedding"}

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """Fetch a document by specifying its uuid string"""
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        index = self._sanitize_index_name(index) or self.index
        document = None

        id = self._sanitize_id(id=id, index=index)
        result = None
        try:
            result = self.weaviate_client.data_object.get_by_id(id, class_name=index, with_vector=True)
        except weaviate.exceptions.UnexpectedStatusCodeException as usce:
            logger.debug("Weaviate could not get the document requested: %s", usce)
        if result:
            # We retrieve the JSON properties from the schema and convert them back to the Python dicts
            json_properties = self._get_json_properties(index=index)
            document = self._convert_weaviate_result_to_document(
                result, return_embedding=True, json_properties=json_properties
            )
        return document

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Fetch documents by specifying a list of uuid strings.
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        index = self._sanitize_index_name(index) or self.index
        batch_size = batch_size or self.batch_size
        # We retrieve the JSON properties from the schema and convert them back to the Python dicts
        json_properties = self._get_json_properties(index=index)
        documents = []
        # TODO: better implementation with multiple where filters instead of chatty call below?
        for id in ids:
            id = self._sanitize_id(id=id, index=index)
            result = None
            try:
                result = self.weaviate_client.data_object.get_by_id(id, class_name=index, with_vector=True)
            except weaviate.exceptions.UnexpectedStatusCodeException as usce:
                logger.debug("Weaviate could not get the document requested: %s", usce)
            if result:
                document = self._convert_weaviate_result_to_document(
                    result, return_embedding=True, json_properties=json_properties
                )
                documents.append(document)
        return documents

    def _sanitize_id(self, id: str, index: Optional[str] = None) -> str:
        """
        Generate a valid uuid if the provided id is not in uuid format.
        Two documents with the same provided id and index name will get the same uuid.
        """
        index = self._sanitize_index_name(index) or self.index
        if not UUID_PATTERN.match(id):
            hashed_id = hashlib.sha256((id + index).encode("utf-8"))  # type: ignore
            generated_uuid = str(uuid.UUID(hashed_id.hexdigest()[::2]))
            if not self.uuid_format_warning_raised:
                logger.warning(
                    "Document id %s is not in uuid format. Such ids will be replaced by uuids, in this case %s.",
                    id,
                    generated_uuid,
                )
                self.uuid_format_warning_raised = True
            id = generated_uuid
        return id

    def _get_current_properties(self, index: Optional[str] = None) -> List[str]:
        """
        Get all the existing properties in the schema, excluding those with complex types
        like cross-reference
        """
        index = self._sanitize_index_name(index) or self.index
        cur_properties = []
        for class_item in self.weaviate_client.schema.get()["classes"]:
            if class_item["class"] == index:
                cur_properties = [
                    item["name"]
                    for item in class_item.get("properties", [])
                    # dataType should be always there and contain only one item unless
                    # it's a cross-reference but here we try to be defensive against
                    # unexpected schemas
                    if set(item.get("dataType", [])).issubset(SUPPORTED_PROPERTY_TYPES)
                ]

        return cur_properties

    def _get_date_properties(self, index: Optional[str] = None) -> List[str]:
        """
        Get all existing properties of type 'date' in the schema.
        """
        index = self._sanitize_index_name(index) or self.index
        cur_properties = []
        for class_item in self.weaviate_client.schema.get()["classes"]:
            if class_item["class"] == index:
                cur_properties = [item["name"] for item in class_item["properties"] if item["dataType"][0] == "date"]

        return cur_properties

    def _get_json_properties(self, index: Optional[str] = None) -> List[str]:
        """
        Get all existing properties that store JSON strings in the indexes' schema.
        """
        index = self._sanitize_index_name(index) or self.index
        cur_properties = []
        for class_item in self.weaviate_client.schema.get()["classes"]:
            if class_item["class"] == index:
                cur_properties = [
                    item["name"]
                    for item in class_item["properties"]
                    if item["description"].startswith("JSON dynamic property")
                ]

        return cur_properties

    def _update_schema(
        self, new_prop: str, property_value: Union[List, str, int, float, bool], index: Optional[str] = None
    ):
        """
        Updates the schema with a new property.
        """
        index = self._sanitize_index_name(index) or self.index
        data_type = self._get_weaviate_type_of_value(property_value)

        description = f"dynamic property {new_prop}"
        if data_type.startswith("dict"):
            # Weaviate does not support dict type, so we use string or list of string instead
            data_type = "string" + data_type[4:]
            description = f"JSON dynamic property {new_prop}"

        property_dict = {"dataType": [data_type], "description": description, "name": new_prop}
        self.weaviate_client.schema.property.create(index, property_dict)

    @staticmethod
    def _get_weaviate_type_of_value(value: Union[List, str, int, float, bool]) -> str:
        """
        Infers corresponding Weaviate data type for a value.
        """
        data_type = ""
        list_of_values = False
        if isinstance(value, list):
            list_of_values = True
            value = value[0]

        if isinstance(value, str):
            # If the value is parsable by datetime, it is a date
            try:
                convert_date_to_rfc3339(value)
                data_type = "date"
            # Otherwise, the value is a string
            except ValueError:
                data_type = "string"
        elif isinstance(value, int):
            data_type = "int"
        elif isinstance(value, float):
            data_type = "number"
        elif isinstance(value, bool):
            data_type = "boolean"
        # Weaviate doesn't support dictionaries as property types, we'll convert them to JSON strings.
        elif isinstance(value, dict):
            data_type = "dict"

        if list_of_values:
            data_type += "[]"

        return data_type

    def _check_document(self, cur_props: List[str], doc: dict) -> List[str]:
        """
        Find the properties in the document that don't exist in the existing schema.
        """
        return [item for item in doc.keys() if item not in cur_props]

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: Optional[int] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. A dummy embedding vector for each document is automatically generated if it is not provided. The document id needs to be in uuid format. Otherwise a correctly formatted uuid will be automatically generated based on the provided id.
        :param index: index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
                           If no batch_size is provided, self.batch_size is used.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        index = self._sanitize_index_name(index) or self.index
        batch_size = batch_size or self.batch_size
        self._create_schema_and_index(index, recreate_index=False)
        field_map = self._create_document_field_map()

        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if len(documents) == 0:
            logger.warning("Calling DocumentStore.write_documents() with empty list")
            return

        # Auto schema feature https://github.com/semi-technologies/weaviate/issues/1539
        # Get and cache current properties in the schema
        current_properties = self._get_current_properties(index)

        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]

        # Weaviate has strict requirements for what ids can be used.
        # We check the id format and sanitize it if no uuid was provided.
        # Duplicate document ids will be mapped to the same generated uuid.
        for do in document_objects:
            do.id = self._sanitize_id(id=do.id, index=index)

        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents
        )

        # Weaviate requires that documents contain a vector in order to be indexed. These lines add a
        # dummy vector so that indexing can still happen
        dummy_embed_warning_raised = False
        for do in document_objects:
            if do.embedding is None:
                dummy_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
                do.embedding = dummy_embedding
                if not dummy_embed_warning_raised:
                    logger.warning(
                        "No embedding found in Document object being written into Weaviate. A dummy "
                        "embedding is being supplied so that indexing can still take place. This "
                        "embedding should be overwritten in order to perform vector similarity searches."
                    )
                    dummy_embed_warning_raised = True

        # get the date properties of the index
        date_fields = self._get_date_properties(index)

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(total=len(document_objects), disable=not self.progress_bar) as progress_bar:
            for document_batch in batched_documents:
                json_fields = []
                for doc in document_batch:
                    _doc = {**doc.to_dict(field_map=self._create_document_field_map())}
                    _ = _doc.pop("score", None)

                    # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
                    # we "unnest" all value within "meta"
                    if "meta" in _doc.keys():
                        for k, v in _doc["meta"].items():
                            if k in _doc.keys():
                                raise ValueError(
                                    f'"meta" info contains duplicate key "{k}" with the top-level document structure'
                                )
                            # Weaviate doesn't support dictionaries as property types, so we convert them to JSON strings
                            if isinstance(v, dict):
                                json_fields.append(k)
                                v = json.dumps(v)
                            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                json_fields.append(k)
                                v = [json.dumps(item) for item in v]
                            _doc[k] = v
                        _doc.pop("meta")

                    doc_id = str(_doc.pop("id"))
                    vector = _doc.pop(self.embedding_field)

                    if self.similarity == "cosine":
                        self.normalize_embedding(vector)

                    # Converting content to JSON-string as Weaviate doesn't allow other nested list for tables
                    _doc["content"] = json.dumps(_doc["content"])

                    # Check if additional properties are in the document, if so,
                    # append the schema with all the additional properties
                    missing_props = self._check_document(current_properties, _doc)
                    if missing_props:
                        for property in missing_props:
                            property_value = _doc[property]
                            if property in json_fields:
                                property_value = doc.meta[property]
                            self._update_schema(property, property_value, index)
                            current_properties.append(property)
                        # update the date fields as there might be new ones
                        date_fields = self._get_date_properties(index)

                    # Weaviate requires dates to be in RFC3339 format
                    for date_field in date_fields:
                        _doc[date_field] = convert_date_to_rfc3339(_doc[date_field])

                    self.weaviate_client.batch.add_data_object(
                        data_object=_doc, class_name=index, uuid=doc_id, vector=vector
                    )
                # Ingest a batch of documents
                results = self.weaviate_client.batch.create_objects()
                # Weaviate returns errors for every failed document in the batch
                if results is not None:
                    for result in results:
                        if (
                            "result" in result
                            and "errors" in result["result"]
                            and "error" in result["result"]["errors"]
                        ):
                            for message in result["result"]["errors"]["error"]:
                                logger.error(message["message"])
                progress_bar.update(batch_size)
        progress_bar.close()

    def update_document_meta(
        self, id: str, meta: Dict[str, Union[List, str, int, float, bool]], index: Optional[str] = None
    ):
        """
        Update the metadata dictionary of a document by specifying its string id.
        Overwrites only the specified fields, the unspecified ones remain unchanged.
        """
        if not index:
            index = self.index

        current_properties = self._get_current_properties(index)

        # Check if the new metadata contains additional properties and append them to the schema
        missing_props = self._check_document(current_properties, meta)
        if missing_props:
            for property in missing_props:
                self._update_schema(property, meta[property], index)
                current_properties.append(property)

        # Weaviate requires dates to be in RFC3339 format
        date_fields = self._get_date_properties(index)
        for date_field in date_fields:
            if date_field in meta and isinstance(meta[date_field], str):
                meta[date_field] = convert_date_to_rfc3339(str(meta[date_field]))

        self.weaviate_client.data_object.update(meta, class_name=index, uuid=id)

    def get_embedding_count(self, filters: Optional[FilterType] = None, index: Optional[str] = None) -> int:
        """
        Return the number of embeddings in the document store, which is the same as the number of documents since
        every document has a default embedding.
        """
        return self.get_document_count(filters=filters, index=index)

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the number of documents in the document store.
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        if only_documents_without_embedding:
            return 0

        index = self._sanitize_index_name(index) or self.index
        doc_count = 0
        if filters:
            filter_dict = LogicalFilterClause.parse(filters).convert_to_weaviate()
            result = self.weaviate_client.query.aggregate(index).with_meta_count().with_where(filter_dict).do()
        else:
            result = self.weaviate_client.query.aggregate(index).with_meta_count().do()

        if "data" in result and "Aggregate" in result.get("data") and result.get("data").get("Aggregate").get(index):
            doc_count = result.get("data").get("Aggregate").get(index)[0]["meta"]["count"]

        return doc_count

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        Note this limitation from the changelog of Weaviate 1.8.0:

        .. quote::
            Due to the increasing cost of each page outlined above, there is a limit to
            how many objects can be retrieved using pagination. By default setting the sum
            of offset and limit to higher than 10,000 objects, will lead to an error.
            If you must retrieve more than 10,000 objects, you can increase this limit by
            setting the environment variable `QUERY_MAXIMUM_RESULTS=<desired-value>`.

            Warning: Setting this to arbitrarily high values can make the memory consumption
            of a single query explode and single queries can slow down the entire cluster.
            We recommend setting this value to the lowest possible value that does not
            interfere with your users' expectations.

        (https://github.com/semi-technologies/weaviate/releases/tag/v1.8.0)

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
                           If no batch_size is provided, self.batch_size is used.
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        index = self._sanitize_index_name(index) or self.index
        batch_size = batch_size or self.batch_size
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def _get_all_documents_in_index(
        self,
        index: Optional[str],
        filters: Optional[FilterType] = None,
        batch_size: Optional[int] = None,
        only_documents_without_embedding: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        index = self._sanitize_index_name(index) or self.index
        batch_size = batch_size or self.batch_size

        # Build the properties to retrieve from Weaviate
        properties = self._get_current_properties(index)
        if self.similarity == "cosine":
            properties.append("_additional {id, certainty, vector}")
        else:
            properties.append("_additional {id, distance, vector}")

        if filters:
            filter_dict = LogicalFilterClause.parse(filters).convert_to_weaviate()
            result = (
                self.weaviate_client.query.get(class_name=index, properties=properties).with_where(filter_dict).do()
            )
        else:
            result = self.weaviate_client.query.get(class_name=index, properties=properties).do()

        # Inherent Weaviate limitation to 100 elements forces us to loop here:
        #   https://weaviate-python-client.readthedocs.io/en/latest/weaviate.data.html?highlight=100#weaviate.data.DataObject.get
        base_query = self.weaviate_client.query.get(class_name=index, properties=properties)
        all_docs: List[Any] = []
        num_of_documents = self.get_document_count(index=index, filters=filters)

        while len(all_docs) < num_of_documents:
            query = base_query
            if filters:
                filter_dict = LogicalFilterClause.parse(filters).convert_to_weaviate()
                query = query.with_where(filter_dict)

            if all_docs:
                # Passing offset:0 raises an error, so we pass it only after the first round
                # `.with_limit()` must be used with `.with_offset`, or the latter won't work properly
                # https://weaviate-python-client.readthedocs.io/en/latest/weaviate.gql.html?highlight=offset#weaviate.gql.get.GetBuilder.with_offset
                query = query.with_limit(100).with_offset(offset=len(all_docs))

            try:
                result = query.do()
            except Exception as e:
                raise WeaviateDocumentStoreError(f"Weaviate raised an exception: {e}")

            if "errors" in result:
                raise WeaviateDocumentStoreError(f"Query results contain errors: {result['errors']}")

            # If `query.do` didn't raise and `result` doesn't contain errors,
            # we are good accessing data
            docs = result.get("data").get("Get").get(index)

            # `docs` can be empty if the query returned less documents than the actual
            # number. This can happen when the number of document stored is greater
            # than QUERY_MAXIMUM_RESULTS.
            # See: https://weaviate.io/developers/weaviate/current/graphql-references/filters.html#offset-argument-pagination
            if not docs:
                logger.warning(
                    "The query returned less documents than expected: this can happen when "
                    "the value of the QUERY_MAXIMUM_RESULTS environment variable is lower than "
                    "the total number of documents stored. See Weaviate documentation for "
                    "more details."
                )
                break

            all_docs += docs

        yield from all_docs

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        Note this limitation from the changelog of Weaviate 1.8.0:

        .. quote::
            Due to the increasing cost of each page outlined above, there is a limit to
            how many objects can be retrieved using pagination. By default setting the sum
            of offset and limit to higher than 10,000 objects, will lead to an error.
            If you must retrieve more than 10,000 objects, you can increase this limit by
            setting the environment variable `QUERY_MAXIMUM_RESULTS=<desired-value>`.

            Warning: Setting this to arbitrarily high values can make the memory consumption
            of a single query explode and single queries can slow down the entire cluster.
            We recommend setting this value to the lowest possible value that does not
            interfere with your users' expectations.

        (https://github.com/semi-technologies/weaviate/releases/tag/v1.8.0)

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
                           If no batch_size is provided, self.batch_size is used.
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        index = self._sanitize_index_name(index) or self.index
        batch_size = batch_size or self.batch_size

        if return_embedding is None:
            return_embedding = self.return_embedding

        results = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size)
        # We retrieve the JSON properties from the schema and convert them back to the Python dicts
        json_properties = self._get_json_properties(index=index)
        for result in results:
            document = self._convert_weaviate_result_to_document(
                result, return_embedding=return_embedding, json_properties=json_properties
            )
            yield document

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
        that are most relevant to the query as defined by Weaviate semantic search.

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
        :param all_terms_must_match: Not used in Weaviate.
        :param custom_query: Custom query that will executed using query.raw method, for more details refer
                            https://weaviate.io/developers/weaviate/current/graphql-references/filters.html
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Not used in Weaviate.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if headers:
            raise NotImplementedError("Weaviate does not support Custom HTTP headers!")

        if all_terms_must_match:
            raise NotImplementedError("The `all_terms_must_match` option is not supported in Weaviate!")

        index = self._sanitize_index_name(index) or self.index

        # Build the properties to retrieve from Weaviate
        properties = self._get_current_properties(index)
        if self.similarity == "cosine":
            properties.append("_additional {id, certainty, vector}")
        else:
            properties.append("_additional {id, distance, vector}")

        if query is None:
            # Retrieval via custom query, no BM25
            if custom_query:
                query_output = self.weaviate_client.query.raw(custom_query)

            # Naive retrieval without BM25, only filtering
            elif filters:
                filter_dict = LogicalFilterClause.parse(filters).convert_to_weaviate()
                query_output = (
                    self.weaviate_client.query.get(class_name=index, properties=properties)
                    .with_where(filter_dict)
                    .with_limit(top_k)
                    .do()
                )
            else:
                raise NotImplementedError(
                    "Weaviate does not support the retrieval of records without specifying a query or a filter!"
                )

        # Default Retrieval via BM25 using the user's query on `self.content_field`
        else:
            # Retrieval with BM25 AND filtering
            if filters:
                filter_dict = LogicalFilterClause.parse(filters).convert_to_weaviate()
                gql_query = (
                    gql.get.GetBuilder(class_name=index, properties=properties, connection=self.weaviate_client)
                    .with_limit(top_k)
                    .with_bm25(query=query, properties=[self.content_field])
                    .with_where(filter_dict)
                    .build()
                )
            else:
                # BM25 retrieval without filtering
                gql_query = (
                    gql.get.GetBuilder(class_name=index, properties=properties, connection=self.weaviate_client)
                    .with_limit(top_k)
                    .with_bm25(query=query, properties=[self.content_field])
                    .build()
                )

            query_output = self.weaviate_client.query.raw(gql_query)

        results = []
        if (
            query_output
            and "data" in query_output
            and "Get" in query_output.get("data")
            and query_output.get("data").get("Get").get(index)
        ):
            results = query_output.get("data").get("Get").get(index)

        # We retrieve the JSON properties from the schema and convert them back to the Python dicts
        json_properties = self._get_json_properties(index=index)
        documents = []
        for result in results:
            doc = self._convert_weaviate_result_to_document(
                result, return_embedding=True, scale_score=scale_score, json_properties=json_properties
            )
            documents.append(doc)

        return documents

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
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the provided queries as defined by keyword matching algorithms like BM25.

        This method lets you find relevant documents for a single query string (output: List of Documents), or a
        a list of query strings (output: List of Lists of Documents).

        :param queries: Single query or list of queries.
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
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        # TODO - This method currently just calls query multiple times. Adapt this once there is a batch querying
        # endpoint in Weaviate, which is currently not available,
        # see https://stackoverflow.com/questions/71558676/does-weaviate-support-bulk-query#comment126569547_71561939

        documents = []

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        # run each query against Weaviate separately and combine the returned documents
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
        :param index: index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :return:
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        index = self._sanitize_index_name(index) or self.index

        # Build the properties to retrieve from Weaviate
        properties = self._get_current_properties(index)
        if self.similarity == "cosine":
            properties.append("_additional {id, certainty, vector}")
        else:
            properties.append("_additional {id, distance, vector}")

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        query_string = {"vector": query_emb}
        if filters:
            filter_dict = LogicalFilterClause.parse(filters).convert_to_weaviate()
            query_output = (
                self.weaviate_client.query.get(class_name=index, properties=properties)
                .with_where(filter_dict)
                .with_near_vector(query_string)
                .with_limit(top_k)
                .do()
            )
        else:
            query_output = (
                self.weaviate_client.query.get(class_name=index, properties=properties)
                .with_near_vector(query_string)
                .with_limit(top_k)
                .do()
            )

        results = []
        if (
            query_output
            and "data" in query_output
            and "Get" in query_output.get("data")
            and query_output.get("data").get("Get").get(index)
        ):
            results = query_output.get("data").get("Get").get(index)

        # We retrieve the JSON properties from the schema and convert them back to the Python dicts
        json_properties = self._get_json_properties(index=index)
        documents = []
        for result in results:
            doc = self._convert_weaviate_result_to_document(
                result, return_embedding=return_embedding, scale_score=scale_score, json_properties=json_properties
            )
            documents.append(doc)

        return documents

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        update_existing_embeddings: bool = True,
        batch_size: Optional[int] = None,
    ):
        """
        Updates the embeddings in the document store using the encoding model specified in the retriever.
        This can be useful if you want to change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Weaviate mandates an embedding while creating the document itself.
        This option must be always true for weaviate and it will update the embeddings for all the documents.
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
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
                           If no batch_size is specified, self.batch_size is used.
        :return: None
        """
        index = self._sanitize_index_name(index) or self.index
        batch_size = batch_size or self.batch_size

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing WeaviateDocumentStore()")

        if update_existing_embeddings:
            logger.info(
                "Updating embeddings for all %s docs ...",
                self.get_document_count(index=index) if logger.level > logging.DEBUG else 0,
            )
        else:
            raise RuntimeError(
                "All the documents in Weaviate store have an embedding by default. Only update is allowed!"
            )

        result = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size)
        # We retrieve the JSON properties from the schema and convert them back to the Python dicts
        json_properties = self._get_json_properties(index=index)

        for result_batch in get_batches_from_generator(result, batch_size):
            document_batch = [
                self._convert_weaviate_result_to_document(hit, return_embedding=False, json_properties=json_properties)
                for hit in result_batch
            ]
            embeddings = retriever.embed_documents(document_batch)
            self._validate_embeddings_shape(
                embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim
            )

            if self.similarity == "cosine":
                self.normalize_embedding(embeddings)

            for doc, emb in zip(document_batch, embeddings):
                # Using update method to only update the embeddings, other properties will be in tact
                self.weaviate_client.data_object.update({}, class_name=index, uuid=doc.id, vector=emb)

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.
        :param index: Index name to delete the document from.
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
        :return: None
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        logger.warning(
            """DEPRECATION WARNINGS:
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
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
                            If filters are provided along with a list of IDs, this method deletes the
                            intersection of the two query results (documents that match the filters and
                            have their ID in the list).
        :return: None
        """
        if headers:
            raise NotImplementedError("WeaviateDocumentStore does not support headers.")

        index = self._sanitize_index_name(index) or self.index

        if not filters and not ids:
            # Delete the existing index, then create an empty new one
            self._create_schema_and_index(index, recreate_index=True)
            return

        # Create index if it doesn't exist yet
        self._create_schema_and_index(index, recreate_index=False)

        if ids and not filters:
            for id in ids:
                self.weaviate_client.data_object.delete(id, class_name=index)

        else:
            # Use filters to restrict list of retrieved documents, before checking these against provided ids
            docs_to_delete = self.get_all_documents(index, filters=filters)
            if ids:
                docs_to_delete = [doc for doc in docs_to_delete if doc.id in ids]
            for doc in docs_to_delete:
                self.weaviate_client.data_object.delete(doc.id, class_name=index)

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        if index == self.index:
            logger.warning(
                "Deletion of default index '%s' detected. "
                "If you plan to use this index again, please reinstantiate '%s' in order to avoid side-effects.",
                index,
                self.__class__.__name__,
            )
        self._delete_index(index)

    def _delete_index(self, index: str):
        index = self._sanitize_index_name(index) or index
        if any(c for c in self.weaviate_client.schema.get()["classes"] if c["class"] == index):
            self.weaviate_client.schema.delete_class(index)
            logger.info("Index '%s' deleted.", index)

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Implemented to respect BaseDocumentStore's contract.

        Weaviate does not support labels (yet).
        """
        raise NotImplementedError("Weaviate does not support labels (yet).")

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        """
        Implemented to respect BaseDocumentStore's contract.

        Weaviate does not support labels (yet).
        """
        raise NotImplementedError("Weaviate does not support labels (yet).")

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Implemented to respect BaseDocumentStore's contract.

        Weaviate does not support labels (yet).
        """
        raise NotImplementedError("Weaviate does not support labels (yet).")

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Implemented to respect BaseDocumentStore's contract.

        Weaviate does not support labels (yet).
        """
        raise NotImplementedError("Weaviate does not support labels (yet).")
