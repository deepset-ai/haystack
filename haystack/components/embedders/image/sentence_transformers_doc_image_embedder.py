# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Any, Dict, List, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _extract_image_sources_info,
    _PDFPageInfo,
)
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackend,
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.device import ComponentDevice
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image


@component
class SentenceTransformersDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Sentence Transformers models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    ### Usage example
    ```python
    from haystack import Document
    from haystack.components.embedders.image import SentenceTransformersDocumentImageEmbedder

    embedder = SentenceTransformersDocumentImageEmbedder(model="sentence-transformers/clip-ViT-B-32")
    embedder.warm_up()

    documents = [
        Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
        Document(content="A photo of a dog", meta={"file_path": "dog.jpg"}),
    ]

    result = embedder.run(documents=documents)
    documents_with_embeddings = result["documents"]
    print(documents_with_embeddings)

    # [Document(id=...,
    #           content='A photo of a cat',
    #           meta={'file_path': 'cat.jpg',
    #                 'embedding_source': {'type': 'image', 'file_path_meta_field': 'file_path'}},
    #           embedding=vector of size 512),
    #  ...]
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        model: str = "sentence-transformers/clip-ViT-B-32",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        encode_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        """
        Creates a SentenceTransformersDocumentEmbedder component.

        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param model:
            The Sentence Transformers model to use for calculating embeddings. Pass a local path or ID of the model on
            Hugging Face. To be used with this component, the model must be able to embed images and text into the same
            vector space. Compatible models include:
            - "sentence-transformers/clip-ViT-B-32"
            - "sentence-transformers/clip-ViT-L-14"
            - "sentence-transformers/clip-ViT-B-16"
            - "sentence-transformers/clip-ViT-B-32-multilingual-v1"
            - "jinaai/jina-embeddings-v4"
            - "jinaai/jina-clip-v1"
            - "jinaai/jina-clip-v2".
        :param device:
            The device to use for loading the model.
            Overrides the default device.
        :param token:
            The API token to download private models from Hugging Face.
        :param batch_size:
            Number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when embedding documents.
        :param normalize_embeddings:
            If `True`, the embeddings are normalized using L2 normalization, so that each embedding has a norm of 1.
        :param trust_remote_code:
            If `False`, allows only Hugging Face verified model architectures.
            If `True`, allows custom models and scripts.
        :param local_files_only:
            If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param config_kwargs:
            Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
        :param precision:
            The precision to use for the embeddings.
            All non-float32 precisions are quantized embeddings.
            Quantized embeddings are smaller and faster to compute, but may have a lower accuracy.
            They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
        :param encode_kwargs:
            Additional keyword arguments for `SentenceTransformer.encode` when embedding documents.
            This parameter is provided for fine customization. Be careful not to clash with already set parameters and
            avoid passing parameters that change the output type.
        :param backend:
            The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
            Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
            for more information on acceleration and quantization options.
        """
        pillow_import.check()

        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.encode_kwargs = encode_kwargs
        self.precision = precision
        self.backend = backend
        self._embedding_backend: Optional[_SentenceTransformersEmbeddingBackend] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            model=self.model,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            precision=self.precision,
            encode_kwargs=self.encode_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDocumentImageEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(init_params, keys=["token"])
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if self._embedding_backend is None:
            self._embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model=self.model,
                device=self.device.to_torch_str(),
                auth_token=self.token,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )
            if self.tokenizer_kwargs and self.tokenizer_kwargs.get("model_max_length"):
                self._embedding_backend.model.max_seq_length = self.tokenizer_kwargs["model_max_length"]

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Embed a list of documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SentenceTransformersDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder."
            )
        if self._embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path
        )

        images_to_embed: List = [None] * len(documents)
        pdf_page_infos: List[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["mime_type"] == "application/pdf":
                # Store PDF documents for later processing
                page_number = image_source_info.get("page_number")
                assert page_number is not None  # checked in _extract_image_sources_info but mypy doesn't know that
                pdf_page_info: _PDFPageInfo = {
                    "doc_idx": doc_idx,
                    "path": image_source_info["path"],
                    "page_number": page_number,
                }
                pdf_page_infos.append(pdf_page_info)
            else:
                # Process images directly
                image = Image.open(image_source_info["path"])
                images_to_embed[doc_idx] = image

        pdf_images_by_doc_idx = _batch_convert_pdf_pages_to_images(pdf_page_infos=pdf_page_infos, return_base64=False)
        for doc_idx, pil_image in pdf_images_by_doc_idx.items():
            images_to_embed[doc_idx] = pil_image

        none_images_doc_ids = [documents[doc_idx].id for doc_idx, image in enumerate(images_to_embed) if image is None]
        if none_images_doc_ids:
            raise RuntimeError(f"Conversion failed for some documents. Document IDs: {none_images_doc_ids}.")

        embeddings = self._embedding_backend.embed(
            data=images_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            precision=self.precision,
            **(self.encode_kwargs if self.encode_kwargs else {}),
        )

        docs_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            copied_doc = copy(doc)
            copied_doc.embedding = emb
            # we store this information for later inspection
            copied_doc.meta["embedding_source"] = {"type": "image", "file_path_meta_field": self.file_path_meta_field}
            docs_with_embeddings.append(copied_doc)

        return {"documents": docs_with_embeddings}
