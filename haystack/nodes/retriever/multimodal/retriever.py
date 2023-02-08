from typing import Union, Optional, Dict, List, Any

import logging
from pathlib import Path

import torch
import numpy as np

from haystack.nodes.retriever import DenseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.schema import ContentTypes, Document, FilterType
from haystack.nodes.retriever.multimodal.embedder import MultiModalEmbedder


logger = logging.getLogger(__name__)


class MultiModalRetriever(DenseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str],
        document_embedding_models: Dict[str, Union[Path, str]],  # Replace str with ContentTypes starting Python3.8
        query_type: str = "text",  # Replace str with ContentTypes starting Python3.8
        query_feature_extractor_params: Optional[Dict[str, Any]] = None,
        document_feature_extractors_params: Optional[Dict[str, Dict[str, Any]]] = None,
        top_k: int = 10,
        batch_size: int = 16,
        embed_meta_fields: Optional[List[str]] = None,
        similarity_function: str = "dot_product",
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        Retriever that uses a multiple encoder to jointly retrieve among a database consisting of different
        data types. See the original paper for more details:
        Kostić, Bogdan, et al. (2021): "Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models"
        (https://arxiv.org/abs/2108.04049),

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
            one used by Hugging Face transformers' modelhub models.
        :param document_embedding_models: Dictionary matching a local path or remote name of document encoder
            checkpoint with the content type it should handle ("text", "table", "image", and so on).
            The format equals the one used by Hugging Face transformers' modelhub models.
        :param query_type: The content type of the query ("text", "image" and so on).
        :param query_feature_extraction_params: The parameters to pass to the feature extractor of the query. If no value is provided, a default dictionary with "max_length": 64 will be set.
        :param document_feature_extraction_params: The parameters to pass to the feature extractor of the documents. If no value is provided, a default dictionary with "text": {"max_length": 256} will be set.
        :param top_k: How many documents to return per query.
        :param batch_size: Number of questions or documents to encode at once. For multiple GPUs, this is
            the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields to a (text) pair that is then used to create
            the embedding. This is likely to improve performance if your titles contain meaningful information
            for retrieval (topic, entities, and so on). Note that only text and table documents support this feature. If no values is provided, a default with "name" as embedding field will be created.
        :param similarity_function: Which function to apply for calculating the similarity of query and document
            embeddings during training. Options: `dot_product` (default) or `cosine`.
        :param progress_bar: Whether to show a tqdm progress bar or not.
            Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices to limit inference to certain GPUs and not use all available ones.
            These strings will be converted into pytorch devices, so use the string notation described in [Tensor Attributes]
            (https://pytorch.org/docs/simage/tensor_attributes.html?highlight=torch%20device#torch.torch.device)
            (e.g. ["cuda:0"]). Note: As multi-GPU training is currently not implemented for TableTextRetriever,
            training only uses the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Hugging Face. If this parameter is set
            to `True`, the local token is used, which must be previously created using `transformer-cli login`.
            For more information, see
            [Hugging Face documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained)
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value
            range are scaled to a range of [0,1], where 1 means extremely relevant.
            Otherwise raw similarity scores (for example, cosine or dot_product) are used.
        """
        if query_feature_extractor_params is None:
            query_feature_extractor_params = {"max_length": 64}
        if document_feature_extractors_params is None:
            document_feature_extractors_params = {"text": {"max_length": 256}}
        if embed_meta_fields is None:
            embed_meta_fields = ["name"]
        super().__init__()

        self.similarity_function = similarity_function
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.scale_score = scale_score

        self.document_embedder = MultiModalEmbedder(
            embedding_models=document_embedding_models,
            feature_extractors_params=document_feature_extractors_params,
            batch_size=batch_size,
            embed_meta_fields=embed_meta_fields,
            progress_bar=progress_bar,
            devices=devices,
            use_auth_token=use_auth_token,
        )

        # Try to reuse the same embedder for queries if there is overlap
        if document_embedding_models.get(query_type, None) == query_embedding_model:
            self.query_embedder = self.document_embedder
        else:
            self.query_embedder = MultiModalEmbedder(
                embedding_models={query_type: query_embedding_model},
                feature_extractors_params={query_type: query_feature_extractor_params},
                batch_size=batch_size,
                embed_meta_fields=embed_meta_fields,
                progress_bar=progress_bar,
                devices=devices,
                use_auth_token=use_auth_token,
            )

        self.document_store = document_store

    def retrieve(  # type: ignore
        self,
        query: Any,
        query_type: ContentTypes = "text",
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number of documents that are most relevant to the
        supplied query. Returns a list of Documents.

        :param query: Query value. It might be text, a path, a table, and so on.
        :param query_type: Type of the query ("text", "table", "image" and so on).
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. It can be a single filter applied to each query or a list of filters
                        (one filter per query).
        :param top_k: How many documents to return per query. Must be > 0.
        :param index: The name of the index in the DocumentStore from which to retrieve documents.
        :param batch_size: Number of queries to embed at a time. Must be > 0.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true, similarity scores (for example, cosine or dot_product) which naturally have a different
                            value range is scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (for example, cosine or dot_product) are used.
        """
        return self.retrieve_batch(
            queries=[query],
            queries_type=query_type,
            filters=[filters],
            top_k=top_k,
            index=index,
            headers=headers,
            batch_size=1,
            scale_score=scale_score,
            document_store=document_store,
        )[0]

    def retrieve_batch(  # type: ignore
        self,
        queries: List[Any],
        queries_type: ContentTypes = "text",
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number of documents that are most relevant to the
        supplied queries. Returns a list of lists of Documents (one list per query).

        This method assumes all queries are of the same data type. Mixed-type query batches (for example one image and one text)
        are currently not supported. Group the queries by type and call `retrieve()` on uniform batches only.

        :param queries: List of query values. They might be text, paths, tables, and so on.
        :param queries_type: Type of the query ("text", "table", "image" and so on)
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. It can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).
        :param top_k: How many documents to return per query. Must be > 0.
        :param index: The name of the index in the DocumentStore from which to retrieve documents.
        :param batch_size: Number of queries to embed at a time. Must be > 0.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If True, similarity scores (for example, cosine or dot_product) which naturally have a different
                            value range are scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (for example, cosine or dot_product) are used.
        """
        top_k = top_k or self.top_k
        document_store = document_store or self.document_store
        if not document_store:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() or retrieve_batch() method."
            )
        index = index or document_store.index
        scale_score = scale_score or self.scale_score

        # Embed the queries - we need them into Document format to leverage MultiModalEmbedder.embed()
        query_docs = [Document(content=query, content_type=queries_type) for query in queries]
        query_embeddings = self.query_embedder.embed(documents=query_docs, batch_size=batch_size)

        # Query documents by embedding (the actual retrieval step)
        documents = document_store.query_by_embedding_batch(
            query_embs=query_embeddings,
            top_k=top_k,
            filters=filters,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )

        return documents

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        return self.document_embedder.embed(documents=docs)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        query_documents = [Document(content=query, content_type="text") for query in queries]
        return self.query_embedder.embed(documents=query_documents)
