import logging
from typing import Any, Dict, Generator, List, Optional, Union, cast

import numpy as np
import qdrant_client
from grpc import RpcError
from haystack import Document, Label
from haystack.document_stores import BaseDocumentStore

try:
    from haystack.utils.batching import get_batches_from_generator
except ImportError:
    from haystack.document_stores.base import get_batches_from_generator

from haystack.errors import DocumentStoreError
from haystack.nodes import DenseRetriever
from haystack.schema import FilterType
from qdrant_client import grpc
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm


from qdrant_haystack.document_stores.converters import (
    HaystackToQdrant,
    QdrantToHaystack,
)
from qdrant_haystack.document_stores.filters import QdrantFilterConverter

logger = logging.getLogger(__name__)


class QdrantStoreError(DocumentStoreError):
    pass


class QdrantDocumentStore(BaseDocumentStore):
    SIMILARITY = {
        "cosine": rest.Distance.COSINE,
        "dot_product": rest.Distance.DOT,
        "l2": rest.Distance.EUCLID,
    }

    def __init__(
            self,
            location: Optional[str] = None,
            url: Optional[str] = None,
            port: int = 6333,
            grpc_port: int = 6334,
            prefer_grpc: bool = False,
            https: Optional[bool] = None,
            api_key: Optional[str] = None,
            prefix: Optional[str] = None,
            timeout: Optional[float] = None,
            host: Optional[str] = None,
            path: Optional[str] = None,
            index: str = "Document",
            embedding_dim: int = 768,
            content_field: str = "content",
            name_field: str = "name",
            embedding_field: str = "vector",
            similarity: str = "cosine",
            return_embedding: bool = False,
            progress_bar: bool = True,
            duplicate_documents: str = "overwrite",
            recreate_index: bool = False,
            shard_number: Optional[int] = None,
            replication_factor: Optional[int] = None,
            write_consistency_factor: Optional[int] = None,
            on_disk_payload: Optional[bool] = None,
            hnsw_config: Optional[dict] = None,
            optimizers_config: Optional[dict] = None,
            wal_config: Optional[dict] = None,
            quantization_config: Optional[dict] = None,
            init_from: Optional[dict] = None,
            wait_result_from_api: bool = True,
            metadata: Optional[dict] = None,
    ):
        super().__init__()

        metadata = metadata or {}
        self.client = qdrant_client.QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            metadata=metadata,
        )

        # Store the Qdrant specific attributes
        self.shard_number = shard_number
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor
        self.on_disk_payload = on_disk_payload
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config
        self.init_from = init_from
        self.wait_result_from_api = wait_result_from_api

        # Make sure the collection is properly set up
        self._set_up_collection(index, embedding_dim, recreate_index, similarity)

        self.embedding_dim = embedding_dim
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.similarity = similarity
        self.index = index
        self.return_embedding = return_embedding
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents
        self.qdrant_filter_converter = QdrantFilterConverter()
        self.haystack_to_qdrant_converter = HaystackToQdrant()
        self.qdrant_to_haystack = QdrantToHaystack(
            content_field,
            name_field,
            embedding_field,
        )

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[FilterType] = None,
            return_embedding: Optional[bool] = None,
            batch_size: int = 10_000,
            headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        return list(
            self.get_all_documents_generator(
                index, filters, return_embedding, batch_size, headers
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
        index = index or self.index
        qdrant_filters = self.qdrant_filter_converter.convert(filters)

        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            records, next_offset = self.client.scroll(
                collection_name=index,
                scroll_filter=qdrant_filters,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            stop_scrolling = next_offset is None or (
                    isinstance(next_offset, grpc.PointId)
                    and next_offset.num == 0
                    and next_offset.uuid == ""
            )

            for record in records:
                yield self.qdrant_to_haystack.point_to_document(record)

    def get_document_by_id(
            self,
            id: str,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Document]:
        documents = self.get_documents_by_id([id], index, headers)
        if 0 == len(documents):
            return None
        return documents[0]

    def get_documents_by_id(
            self,
            ids: List[str],
            index: Optional[str] = None,
            batch_size: int = 10_000,
            headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        index = index or self.index

        documents: List[Document] = []

        next_offset = None
        stop_scrolling = False
        scroll_filter = self.qdrant_filter_converter.convert(None, ids)
        while not stop_scrolling:
            records, next_offset = self.client.scroll(
                collection_name=index,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            stop_scrolling = next_offset is None or (
                    isinstance(next_offset, grpc.PointId)
                    and next_offset.num == 0
                    and next_offset.uuid == ""
            )

            for record in records:
                documents.append(self.qdrant_to_haystack.point_to_document(record))

        return documents

    def get_document_count(
            self,
            filters: Optional[FilterType] = None,
            index: Optional[str] = None,
            only_documents_without_embedding: bool = False,
            headers: Optional[Dict[str, str]] = None,
    ) -> int:
        index = index or self.index
        qdrant_filters = self.qdrant_filter_converter.convert(filters)

        try:
            response = self.client.count(
                collection_name=index,
                count_filter=qdrant_filters,
            )
            return response.count
        except (UnexpectedResponse, ValueError):
            # Qdrant local raises ValueError if the collection is not found, but
            # with the remote server UnexpectedResponse is raised. Until that's unified,
            # we need to catch both.
            return 0

    def get_embedding_count(
            self, filters: Optional[FilterType] = None, index: Optional[str] = None
    ) -> int:
        """
        Return the number of embeddings in the document store, which is the same as the
        number of documents since every document has a default embedding.
        """
        return self.get_document_count(filters, index)

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
        index = index or self.index
        qdrant_filters = self.qdrant_filter_converter.convert(filters)

        points = self.client.search(
            collection_name=index,
            query_vector=cast(list, query_emb.tolist()),
            query_filter=qdrant_filters,
            limit=top_k,
            with_vectors=return_embedding or True,
        )

        results = [self.qdrant_to_haystack.point_to_document(point) for point in points]
        if scale_score:
            for document in results:
                document.score = self.scale_to_unit_interval(
                    document.score, self.similarity
                )
        return results

    def write_documents(
            self,
            documents: Union[List[dict], List[Document]],
            index: Optional[str] = None,
            batch_size: int = 10_000,
            duplicate_documents: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        index = index or self.index
        self._set_up_collection(index, self.embedding_dim, False, self.similarity)
        field_map = self._create_document_field_map()

        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
                duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if len(documents) == 0:
            logger.warning(
                "Calling QdrantDocumentStore.write_documents() with empty list"
            )
            return

        document_objects = [
            Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d
            for d in documents
        ]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects,
            index=index,
            duplicate_documents=duplicate_documents,
        )

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(
                total=len(document_objects), disable=not self.progress_bar
        ) as progress_bar:
            for document_batch in batched_documents:
                batch = self.haystack_to_qdrant_converter.documents_to_batch(
                    document_batch,
                    embedding_field=self.embedding_field,
                    embedding_dim=self.embedding_dim,
                    field_map=self._create_document_field_map(),
                    fill_missing_embeddings=True,
                )

                response = self.client.upsert(
                    collection_name=index,
                    points=batch,
                    wait=self.wait_result_from_api,
                )

                # TODO: handle errors in response
                progress_bar.update(batch_size)

    def update_embeddings(
            self,
            retriever: DenseRetriever,
            index: Optional[str] = None,
            update_existing_embeddings: bool = True,
            filters: Optional[FilterType] = None,
            batch_size: int = 32,
            headers: Optional[Dict[str, str]] = None,
    ):
        """
        :param retriever:
        :param index:
        :param update_existing_embeddings: Not used by QdrantDocumentStore, as all the points
                                           must have a corresponding vector in Qdrant.
        :param filters:
        :param batch_size:
        :param headers:
        :return:
        """
        index = index or self.index

        document_count = self.get_document_count(index=index, filters=filters)
        if document_count == 0:
            logger.warning(
                "Calling DocumentStore.update_embeddings() on an empty index"
            )
            return

        logger.info("Updating embeddings for %s docs...", document_count)

        doc_generator = self.get_all_documents_generator(
            index=index,
            filters=filters,
            batch_size=batch_size,
            headers=headers,
        )

        with tqdm(
                total=document_count, position=0, unit=" Docs", desc="Updating embeddings"
        ) as progress_bar:
            for document_batch in get_batches_from_generator(doc_generator, batch_size):
                embeddings = retriever.embed_documents(document_batch)
                self._validate_embeddings_shape(
                    embeddings=embeddings,
                    num_documents=len(document_batch),
                    embedding_dim=self.embedding_dim,
                )

                # Overwrite the existing embeddings in that batch
                for doc, embedding in zip(document_batch, embeddings):
                    doc.embedding = embedding

                # Upsert points into Qdrant and overwrite the entries
                batch = self.haystack_to_qdrant_converter.documents_to_batch(
                    document_batch,
                    embedding_field=self.embedding_field,
                    embedding_dim=self.embedding_dim,
                    field_map=self._create_document_field_map(),
                )
                response = self.client.upsert(
                    collection_name=index,
                    points=batch,
                )

                progress_bar.update(batch_size)

    def update_document_meta(
            self, id: str, meta: Dict[str, Any], index: Optional[str] = None
    ):
        index = index or self.index
        document = self.get_document_by_id(id, index)
        if document is None:
            logger.warning(
                "Requested to update document meta for non-existing id %s", id
            )
            return

        document.meta = meta

        # Upsert point into Qdrant and overwrite the entry. Batch is used to keep
        # the same logic as for .update_embeddings.
        batch = self.haystack_to_qdrant_converter.documents_to_batch(
            [document],
            embedding_field=self.embedding_field,
            embedding_dim=self.embedding_dim,
            field_map=self._create_document_field_map(),
        )
        self.client.upsert(
            collection_name=index,
            points=batch,
            wait=self.wait_result_from_api,
        )

    def delete_documents(
            self,
            index: Optional[str] = None,
            ids: Optional[List[str]] = None,
            filters: Optional[FilterType] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        index = index or self.index
        qdrant_filters = self.qdrant_filter_converter.convert(filters, ids)

        self.client.delete(
            collection_name=index,
            points_selector=qdrant_filters,
            wait=self.wait_result_from_api,
        )

    def delete_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[FilterType] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        index = index or self.index
        qdrant_filters = self.qdrant_filter_converter.convert(filters)

        self.client.delete(
            collection_name=index,
            points_selector=qdrant_filters,
            wait=self.wait_result_from_api,
        )

    def delete_index(self, index: str):
        self.client.delete_collection(collection_name=index)

    def get_all_labels(
            self,
            index: Optional[str] = None,
            filters: Optional[FilterType] = None,
            headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        raise NotImplementedError("Qdrant does not support labels yet")

    def get_label_count(
            self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> int:
        raise NotImplementedError("Qdrant does not support labels yet")

    def write_labels(
            self,
            labels: Union[List[Label], List[dict]],
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("Qdrant does not support labels yet")

    def delete_labels(
            self,
            index: Optional[str] = None,
            ids: Optional[List[str]] = None,
            filters: Optional[FilterType] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("Qdrant does not support labels yet")

    def _create_document_field_map(self) -> Dict:
        return {
            self.name_field: "name",
            self.content_field: "content",
            self.embedding_field: "embedding",
        }

    def _get_distance(self, similarity: str) -> rest.Distance:
        try:
            return self.SIMILARITY[similarity]
        except KeyError:
            raise QdrantStoreError(
                f"Provided similarity '{similarity}' is not supported by Qdrant "
                f"document store. Please choose one of the options: "
                f"{', '.join(self.SIMILARITY.keys())}"
            )

    def _set_up_collection(
            self,
            collection_name: str,
            embedding_dim: int,
            recreate_collection: bool,
            similarity: str,
    ):
        distance = self._get_distance(similarity)

        if recreate_collection:
            # There is no need to verify the current configuration of that
            # collection. It might be just recreated again.
            self._recreate_collection(collection_name, distance, embedding_dim)
            return

        try:
            # Check if the collection already exists and validate its
            # current configuration with the parameters.
            collection_info = self.client.get_collection(collection_name)
        except (UnexpectedResponse, RpcError, ValueError):
            # That indicates the collection does not exist, so it can be
            # safely created with any configuration.
            #
            # Qdrant local raises ValueError if the collection is not found, but
            # with the remote server UnexpectedResponse / RpcError is raised.
            # Until that's unified, we need to catch both.
            self._recreate_collection(collection_name, distance, embedding_dim)
            return

        current_distance = collection_info.config.params.vectors.distance
        current_vector_size = collection_info.config.params.vectors.size

        if current_distance != distance:
            raise ValueError(
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it is configured with a similarity '{current_distance.name}'. "
                f"If you want to use that collection, but with a different "
                f"similarity, please set `recreate_collection=True` argument."
            )

        if current_vector_size != embedding_dim:
            raise ValueError(
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it is configured with a vector size '{current_vector_size}'. "
                f"If you want to use that collection, but with a different "
                f"vector size, please set `recreate_collection=True` argument."
            )

    def _recreate_collection(self, collection_name: str, distance, embedding_dim: int):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=embedding_dim,
                distance=distance,
            ),
            shard_number=self.shard_number,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=self.on_disk_payload,
            hnsw_config=self.hnsw_config,
            optimizers_config=self.optimizers_config,
            wal_config=self.wal_config,
            quantization_config=self.quantization_config,
            init_from=self.init_from,
        )
