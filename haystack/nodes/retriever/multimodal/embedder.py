from typing import Union, Optional, Dict, List, Any

import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

from haystack.modeling.model.multimodal import get_model
from haystack.errors import NodeError, ModelingError
from haystack.modeling.model.multimodal.base import HaystackModel
from haystack.schema import Document
from haystack.utils.torch_utils import get_devices


logger = logging.getLogger(__name__)


class MultiModalRetrieverError(NodeError):
    pass


FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


# TODO the keys should match with ContentTypes (currently 'audio' is missing)
DOCUMENT_CONVERTERS = {
    # NOTE: Keep this '?' cleaning step, it needs to be double-checked for impact on the inference results.
    "text": lambda doc: doc.content[:-1] if doc.content[-1] == "?" else doc.content,
    "table": lambda doc: " ".join(
        doc.content.columns.tolist() + [cell for row in doc.content.values.tolist() for cell in row]
    ),
    "image": lambda doc: Image.open(doc.content),
}

CAN_EMBED_META = ["text", "table"]


class MultiModalEmbedder:
    def __init__(
        self,
        embedding_models: Dict[str, Union[Path, str]],  # replace str with ContentTypes starting from Python3.8
        feature_extractors_params: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name"],
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Init the Retriever and all its models from a local or remote model checkpoint.
        The checkpoint format matches the Hugging Face transformers' model format.

        :param embedding_models: A dictionary matching a local path or remote name of encoder checkpoint with
            the content type it should handle ("text", "table", "image", etc...).
            The format is the one that Hugging Face Hub models use.
            Expected input format: `{'text': 'name_or_path_to_text_model', 'image': 'name_or_path_to_image_model', ... }`
            Keep in mind that the models should output in the same embedding space for this retriever to work.
        :param feature_extractors_params: A dictionary matching a content type ("text", "table", "image" and so on) with the
            parameters of its own feature extractor if the model requires one.
            Expected input format: `{'text': {'param_name': 'param_value', ...}, 'image': {'param_name': 'param_value', ...}, ...}`
        :param batch_size: Number of questions or passages to encode at once. In case of multiple GPUs, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / image to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices to limit inference to certain GPUs and not use all available ones.
                        These strings are converted into pytorch devices, so use the string notation described
                        [in the pytorch documentation](https://pytorch.org/docs/simage/tensor_attributes.html?highlight=torch%20device#torch.torch.device)
                        (for example, ["cuda:0"]). Note: as multi-GPU training is currently not implemented for TableTextRetriever,
                        training only uses the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Hugging Face. If this parameter is set to `True`,
                                the local token is used, which must be previously created using `transformer-cli login`.
                                For more information, see [Hugging Face documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained)
        """
        super().__init__()

        self.devices = get_devices(devices)
        if batch_size < len(self.devices):
            logger.warning("Batch size is lower than the number of devices. Not all GPUs will be utilized.")

        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.embed_meta_fields = embed_meta_fields

        feature_extractors_params = {
            content_type: {"max_length": 256, **(feature_extractors_params or {}).get(content_type, {})}
            for content_type in ["text", "table", "image", "audio"]  # FIXME get_args(ContentTypes) from Python3.8 on
        }

        self.models: Dict[str, HaystackModel] = {}  # replace str with ContentTypes starting from Python3.8
        for content_type, embedding_model in embedding_models.items():
            self.models[content_type] = get_model(
                pretrained_model_name_or_path=embedding_model,
                content_type=content_type,
                devices=self.devices,
                autoconfig_kwargs={"use_auth_token": use_auth_token},
                model_kwargs={"use_auth_token": use_auth_token},
                feature_extractor_kwargs=feature_extractors_params[content_type],
            )

        # Check embedding sizes for models: they must all match
        if len(self.models) > 1:
            sizes = {model.embedding_dim for model in self.models.values()}
            if None in sizes:
                logger.warning(
                    "Haystack could not find the output embedding dimensions for '%s'. "
                    "Dimensions won't be checked before computing the embeddings.",
                    ", ".join(
                        {str(model.model_name_or_path) for model in self.models.values() if model.embedding_dim is None}
                    ),
                )
            elif len(sizes) > 1:
                embedding_sizes: Dict[int, List[str]] = {}
                for model in self.models.values():
                    embedding_sizes[model.embedding_dim] = embedding_sizes.get(model.embedding_dim, []) + [
                        str(model.model_name_or_path)
                    ]
                raise ValueError(f"Not all models have the same embedding size: {embedding_sizes}")

    def embed(self, documents: List[Document], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Create embeddings for a list of documents using the relevant encoder for their content type.

        :param documents: Documents to embed.
        :return: Embeddings, one per document, in the form of a np.array
        """
        batch_size = batch_size if batch_size is not None else self.batch_size

        all_embeddings = []
        for batch_index in tqdm(
            iterable=range(0, len(documents), batch_size),
            unit=" Docs",
            desc=f"Create embeddings",
            position=1,
            leave=False,
            disable=not self.progress_bar,
        ):
            docs_batch = documents[batch_index : batch_index + batch_size]
            data_by_type = self._docs_to_data(documents=docs_batch)

            # Get output for each model
            outputs_by_type: Dict[str, torch.Tensor] = {}  # replace str with ContentTypes starting Python3.8
            for data_type, data in data_by_type.items():

                model = self.models.get(data_type)
                if not model:
                    raise ModelingError(
                        f"Some data of type {data_type} was passed, but no model capable of handling such data was "
                        f"initialized. Initialized models: {', '.join(self.models.keys())}"
                    )
                outputs_by_type[data_type] = model.encode(data=data)

            # Check the output sizes
            embedding_sizes = [output.shape[-1] for output in outputs_by_type.values()]

            if not all(embedding_size == embedding_sizes[0] for embedding_size in embedding_sizes):
                raise ModelingError(
                    "Some of the models are using a different embedding size. They should all match. "
                    f"Embedding sizes by model: "
                    f"{ {name: output.shape[-1] for name, output in outputs_by_type.items()} }"
                )

            # Combine the outputs in a single matrix
            outputs = torch.stack(list(outputs_by_type.values()))
            embeddings = outputs.view(-1, embedding_sizes[0])
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings)

    def _docs_to_data(
        self, documents: List[Document]
    ) -> Dict[str, List[Any]]:  # FIXME replace str to ContentTypes from Python3.8
        """
        Extract the data to embed from each document and return them classified by content type.

        :param documents: The documents to prepare fur multimodal embedding.
        :return: A dictionary containing one key for each content type, and a list of data extracted
            from each document, ready to be passed to the feature extractor (for example the content
            of a text document, a linearized table, a PIL image object, and so on)
        """
        docs_data: Dict[str, List[Any]] = {  # FIXME replace str to ContentTypes from Python3.8
            key: [] for key in ["text", "table", "image", "audio"]
        }  # FIXME get_args(ContentTypes) from Python3.8 on
        for doc in documents:
            try:
                document_converter = DOCUMENT_CONVERTERS[doc.content_type]
            except KeyError as e:
                raise MultiModalRetrieverError(
                    f"Unknown content type '{doc.content_type}'. Known types: 'text', 'table', 'image'."  # FIXME {', '.join(get_args(ContentTypes))}"  from Python3.8 on
                ) from e

            data = document_converter(doc)

            if doc.content_type in CAN_EMBED_META:
                meta = [v for k, v in (doc.meta or {}).items() if k in self.embed_meta_fields]
                data = f"{' '.join(meta)} {data}" if meta else data

            docs_data[doc.content_type].append(data)

        return {key: values for key, values in docs_data.items() if values}
