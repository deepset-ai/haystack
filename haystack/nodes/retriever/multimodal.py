from typing import Iterable, get_args, Union, Optional, Dict, List, Any

import logging
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.nn import DataParallel

from haystack.nodes.retriever import BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.modeling.model.multiadaptive_model import MultiAdaptiveModel
from haystack.modeling.model.multimodal_language_model import get_mm_language_model
from haystack.modeling.model.feature_extraction import FeatureExtractor
from haystack.errors import NodeError, ModelingError
from haystack.schema import ContentTypes, Document
from haystack.modeling.data_handler.multimodal_samples.text import TextSample

from haystack.modeling.data_handler.multimodal_samples.base import Sample
from haystack.modeling.data_handler.multimodal_samples.image import ImageSample
from haystack.modeling.model.feature_extraction import FeatureExtractor


logger = logging.getLogger(__name__)


class MultiModalRetrieverError(NodeError):
    pass


DOCUMENT_CONVERTERS = {
    # NOTE: Keep this '?' cleaning step, it needs to be double-checked for impact on the inference results.
    "text": lambda doc: doc.content[:-1] if doc.content[-1] == "?" else doc.content,
    "table": lambda doc: " ".join(
        doc.content.columns.tolist() + [cell for row in doc.content.values.tolist() for cell in row]
    ),
    "image": lambda doc: np.array(Image.open(doc.content).convert("RGB")),
}

CAN_EMBED_META = ["text", "table"]

SAMPLES_BY_DATATYPE: Dict[ContentTypes, Sample] = {"text": TextSample, "table": TextSample, "image": ImageSample}


def get_features(
    data: List[Any],
    data_type: ContentTypes,
    feature_extractor: FeatureExtractor,
    extraction_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return proper features by data type by leveraging Sample classes.
    """
    try:
        sample_class = SAMPLES_BY_DATATYPE[data_type]
    except KeyError as e:
        raise ModelingError(
            f"Data type '{data_type}' not recognized. "
            f"Please select one data type among {', '.join(SAMPLES_BY_DATATYPE.keys())}"
        )
    return sample_class.get_features(
        data=data, feature_extractor=feature_extractor, extraction_params=extraction_params
    )


def get_devices(devices: List[Union[str, torch.device]]) -> List[torch.device]:
    """
    Convert a list of device names into a list of Torch devices,
    depending on the system's configuration and hardware.
    """
    if devices is not None:
        return [torch.device(device) for device in devices]
    elif torch.cuda.is_available():
        return [torch.device(device) for device in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def flatten(iterable: Any):
    """
    Flatten an arbitrarily nested list. Does not unpack tuples or other Iterables.
    Yields a generator. Use `list()` to compute the full list.

    >> list(flatten([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten([[1, 2], 3]))
    [1, 2, 3]
    """
    if isinstance(iterable, list):
        for item in iterable:
            yield from flatten(item)
    else:
        yield (iterable)


class MultiModalEmbedder:
    def __init__(
        self,
        embedding_models: Dict[ContentTypes, Union[Path, str]] = {"text": "facebook/data2vec-text-base"},
        feature_extractors_params: Dict[str, Dict[str, Any]] = None,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name"],
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Init the Retriever and all its models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format.

        :param embedding_models: Dictionary matching a local path or remote name of encoder checkpoint with
            the content type it should handle ("text", "table", "image", etc...).
            The format equals the one used by hugging-face transformers' modelhub models.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / image to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/simage/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: as multi-GPU training is currently not implemented for TableTextRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        self.devices = get_devices(devices)
        if batch_size < len(self.devices):
            logger.warning("Batch size is lower than the number of devices. Not all GPUs will be utilized.")

        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.embed_meta_fields = embed_meta_fields

        self.feature_extractors_params = {
            content_type: {"max_length": 256} | (feature_extractors_params or {}).get(content_type, {})
            for content_type in get_args(ContentTypes)
        }

        self.feature_extractors = {}
        models = {}
        for content_type, embedding_model in embedding_models.items():
            self.feature_extractors[content_type] = FeatureExtractor(
                pretrained_model_name_or_path=embedding_model, do_lower_case=True, use_auth_token=use_auth_token
            )
            models[content_type] = get_mm_language_model(
                pretrained_model_name_or_path=embedding_model,
                content_type=content_type,
                autoconfig_kwargs={"use_auth_token": use_auth_token},
            )

        self.model = MultiAdaptiveModel(models=models, device=self.devices[0])

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)

    def docs_to_data(self, documents: List[Document]) -> Dict[ContentTypes, List[Any]]:
        """
        Extract the data to embed from each document and returns them classified by content type.

        :param documents: the documents to prepare fur multimodal embedding.
        :return: a dictionary containing one key for each content type, and a list of data extracted
            from each document, ready to be passed to the feature extractor (for example the content
            of a text document, a linearized table, a PIL image object, etc...)
        """
        docs_data = {key: [] for key in get_args(ContentTypes)}
        for doc in documents:
            try:
                document_converter = DOCUMENT_CONVERTERS[doc.content_type]
            except KeyError as e:
                raise MultiModalRetrieverError(
                    f"Unknown content type '{doc.content_type}'. Known types: {', '.join(get_args(ContentTypes))}"
                ) from e

            data = document_converter(doc)

            if self.embed_meta_fields and doc.content_type in CAN_EMBED_META:
                meta = " ".join(doc.meta or [])
                docs_data[doc.content_type].append(
                    (meta, data)
                )  # They used to be returned as a tuple, verify it still works as intended
            else:
                docs_data[doc.content_type].append(data)

        return {key: values for key, values in docs_data.items() if values}

    def embed(self, documents: List[Document], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Create embeddings for a list of documents using the relevant encoder for their content type.

        :param documents: Documents to embed
        :return: Embeddings, one per document, in the form of a np.array
        """
        all_embeddings = []

        batch_size = batch_size if batch_size is not None else self.batch_size
        batched_docs = [
            documents[batch_index : batch_index + batch_size] for batch_index in range(0, len(documents), batch_size)
        ]

        with tqdm(
            total=len(documents),
            unit=" Docs",
            desc=f"Create embeddings",
            position=1,
            leave=False,
            disable=not self.progress_bar,
        ) as progress_bar:

            for docs_batch in batched_docs:

                data_by_type = self.docs_to_data(documents=docs_batch)
                if set(data_by_type.keys()) > set(self.feature_extractors.keys()):
                    raise ModelingError(
                        "You provided documents for which you have no embedding model. "
                        "Please provide a suitable embedding model for each document type.\n"
                        f"Detected document types: {', '.join(data_by_type.keys())}\n"
                        f"Embedding model types: {', '.join(self.feature_extractors.keys())}\n"
                    )

                features_by_type = {}
                for data_type, data_list in data_by_type.items():

                    # extract the features in bulk
                    features = get_features(
                        data=data_list,
                        data_type=data_type,
                        feature_extractor=self.feature_extractors[data_type],
                        extraction_params=self.feature_extractors_params.get(data_type, {}),
                    )
                    if not features:
                        raise ModelingError(
                            f"Could not extract features for data of type {data_type}. "
                            f"Check that your feature extractor is correct for this data type:\n{self.feature_extractors}"
                        )
                    features_by_type[data_type] = features

                # Sanity check: the data must have this shape
                # features_by_type = {
                #   "text": {
                #       "input_ids" : [
                #           <tensor>,
                #           <tensor>,
                #           <tensor>,
                #           <tensor>,
                #           ...
                #       ],  # 2d tensor, each row is a document embedding of the type specified in the first level
                #       ...
                #   },
                #   ...
                # }
                assert len(docs_batch) == sum(
                    [list(tensors_by_type.values())[0].shape[0] for tensors_by_type in features_by_type.values()]
                )

                # Get logits
                with torch.no_grad():
                    embeddings = self.model.forward(inputs_by_model=features_by_type)
                    embeddings = embeddings.cpu().numpy()

                all_embeddings.append(embeddings)
                progress_bar.update(batch_size)

        return np.concatenate(all_embeddings)


FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


class MultiModalRetriever(BaseRetriever):
    """
    Retriever that uses a multiple encoder to jointly retrieve among a database consisting of different
    data types. See the original paper for more details:
    Kostić, Bogdan, et al. (2021): "Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models"
    (https://arxiv.org/abs/2108.04049),
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_type: ContentTypes = "text",
        query_embedding_model: Union[Path, str] = "facebook/data2vec-text-base",
        passage_embedding_models: Dict[ContentTypes, Union[Path, str]] = {"text": "facebook/data2vec-text-base"},
        query_feature_extractor_params: Dict[str, Any] = {"max_length": 64},
        passage_feature_extractors_params: Dict[str, Dict[str, Any]] = {"max_length": 256},
        top_k: int = 10,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name"],
        similarity_function: str = "dot_product",
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        Init the Retriever and all its models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format.

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models.
        :param passage_embedding_models: Dictionary matching a local path or remote name of passage encoder checkpoint with
            the content type it should handle ("text", "table", "image", etc...).
            The format equals the one used by hugging-face transformers' modelhub models.
        :param max_seq_len_query:Longest length of each passage/context sequence. Represents the maximum number of tokens for the passage text.
            Longer ones will be cut down.
        :param max_seq_len_passages: Dictionary matching the longest length of each query sequence with the content_type they refer to.
            Represents the maximum number of tokens. Longer ones will be cut down.
        :param top_k: How many documents to return per query.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / image to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/simage/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: as multi-GPU training is currently not implemented for TableTextRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        super().__init__()

        self.similarity_function = similarity_function
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.scale_score = scale_score

        self.query_embedder = MultiModalEmbedder(
            embedding_models={query_type: query_embedding_model},
            feature_extractors_params={query_type: query_feature_extractor_params},
            batch_size=batch_size,
            embed_meta_fields=embed_meta_fields,
            progress_bar=progress_bar,
            devices=devices,
            use_auth_token=use_auth_token,
        )
        self.passage_embedder = MultiModalEmbedder(
            embedding_models=passage_embedding_models,
            feature_extractors_params=passage_feature_extractors_params,
            batch_size=batch_size,
            embed_meta_fields=embed_meta_fields,
            progress_bar=progress_bar,
            devices=devices,
            use_auth_token=use_auth_token,
        )

        self.document_store = document_store

    def retrieve(
        self,
        query: str,
        content_type: ContentTypes = "text",
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ) -> List[Document]:
        return self.retrieve_batch(
            queries=[query],
            content_type=content_type,
            filters=[filters],
            top_k=top_k,
            index=index,
            headers=headers,
            batch_size=1,
            scale_score=scale_score,
        )[0]

    def retrieve_batch(
        self,
        queries: List[str],
        content_type: ContentTypes = "text",
        filters: Optional[Union[FilterType, List[FilterType]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one list per query).

        This method assumes all queries are of the same data type. Mixed-type query batches (i.e. one image and one text)
        are currently not supported. Please group the queries by type and call `retrieve()` on uniform batches only.

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).
        :param top_k: How many documents to return per query. Must be > 0
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time. Must be > 0
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if not isinstance(filters, Iterable):
            filters = [filters or {}] * len(queries)

        elif len(filters) != len(queries):
            raise MultiModalRetrieverError(
                "Number of filters does not match number of queries. Please provide as many filters "
                "as queries, or a single filter that will be applied to all queries."
            )

        top_k = top_k or self.top_k
        index = index or self.document_store.index
        scale_score = scale_score or self.scale_score

        # Embed the queries - we need them into Document format to leverage MultiModalEmbedder.embed()
        query_docs = [Document(content=query, content_type=content_type) for query in queries]
        query_embeddings = self.query_embedder.embed(documents=query_docs, batch_size=batch_size)

        # Query documents by embedding (the actual retrieval step)
        documents = []
        for query_embedding, query_filters in zip(query_embeddings, filters):
            docs = self.document_store.query_by_embedding(
                query_emb=query_embedding,
                top_k=top_k,
                filters=query_filters,
                index=index,
                headers=headers,
                scale_score=scale_score,
            )
            documents.append(docs)
        return documents

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        return self.passage_embedder.embed(documents=docs)
