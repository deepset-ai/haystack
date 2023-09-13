from typing import List, Optional, Union

import logging

from tqdm import tqdm

from haystack.schema import Document
from haystack.nodes.image_to_text.base import BaseImageToText
from haystack.errors import ImageToTextError
from haystack.lazy_imports import LazyImport


logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from transformers import pipeline
    from haystack.modeling.utils import initialize_device_settings  # pylint: disable=ungrouped-imports
    from haystack.utils.torch_utils import ListDataset


# supported models classes should be extended when HF image-to-text pipeline will support more classes
# see https://github.com/huggingface/transformers/issues/21110
SUPPORTED_MODELS_CLASSES = [
    "VisionEncoderDecoderModel",
    "BlipForConditionalGeneration",
    "Blip2ForConditionalGeneration",
]

UNSUPPORTED_MODEL_MESSAGE = (
    f"The supported classes are: {SUPPORTED_MODELS_CLASSES}. \n"
    f"To find the supported models: \n"
    f"1. Visit [image-to-text models on Hugging Face](https://huggingface.co/models?pipeline_tag=image-to-text). \n"
    f"2. Open a model you want to check.  \n"
    f'3. On the model page, go to the "Files and Versions" tab \n'
    f"4. Open the `config.json` file, and make sure the `architectures` field contains one of the supported classes: {SUPPORTED_MODELS_CLASSES}."
)


class TransformersImageToText(BaseImageToText):
    """
    A transformer-based model to generate captions for images using the Hugging Face's transformers framework.

    **Example**

     ```python
        image_file_paths = ["/path/to/images/apple.jpg",
                            "/path/to/images/cat.jpg", ]

        # Generate captions
        documents = image_to_text.generate_captions(image_file_paths=image_file_paths)

        # Show results (List of Documents, containing caption and image file_path)
        print(documents)

        [
            {
                "content": "a red apple is sitting on a pile of hay",
                ...
                "meta": {
                            "image_path": "/path/to/images/apple.jpg",
                            ...
                        },
                ...
            },
            ...
        ]
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "Salesforce/blip-image-captioning-base",
        model_version: Optional[str] = None,
        generation_kwargs: Optional[dict] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
    ):
        """
        Load an Image-to-Text model from transformers.

        :param model_name_or_path: Directory of a saved model or the name of a public model.
                                   To find these models:
                                   1. Visit [Hugging Face image to text models](https://huggingface.co/models?pipeline_tag=image-to-text).`
                                   2. Open the model you want to check.
                                   3. On the model page, go to the "Files and Versions" tab.
                                   4. Open the `config.json` file and make sure the `architectures` field contains `VisionEncoderDecoderModel`, `BlipForConditionalGeneration`, or `Blip2ForConditionalGeneration`.
        :param model_version: The version of the model to use from the Hugging Face model hub. This can be the tag name, branch name, or commit hash.
        :param generation_kwargs: Dictionary containing arguments for the `generate()` method of the Hugging Face model.
                                See [generate()](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate) in Hugging Face documentation.
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of documents to process at a time.
        :param progress_bar: Whether to show a progress bar.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                               If set to `True`, the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) is used.
                               For more information, see [from_pretrained()](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained) in Hugging Face documentation.
        :param devices: List of torch devices (for example, cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects or strings is supported (for example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). If you set `use_gpu=False`, the devices
                        parameter is not used and a single CPU device is used for inference.
        """
        torch_and_transformers_import.check()
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        try:
            self.model = pipeline(
                task="image-to-text",
                model=model_name_or_path,
                revision=model_version,
                device=self.devices[0],
                use_auth_token=use_auth_token,
            )
        except KeyError as err:
            raise ValueError(
                f"The model '{model_name_or_path}' is not supported for ImageToText. " f"{UNSUPPORTED_MODEL_MESSAGE}"
            ) from err

        # for some unsupported models, initializing the HF pipeline doesn't raise errors but does not work
        model_class_name = self.model.model.__class__.__name__
        if model_class_name not in SUPPORTED_MODELS_CLASSES:
            raise ValueError(
                f"The model '{model_name_or_path}' (class '{model_class_name}') is not supported for ImageToText. "
                f"{UNSUPPORTED_MODEL_MESSAGE}"
            )

        self.generation_kwargs = generation_kwargs
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def generate_captions(
        self, image_file_paths: List[str], generation_kwargs: Optional[dict] = None, batch_size: Optional[int] = None
    ) -> List[Document]:
        """
        Generate captions for the image files you specify.

        :param image_file_paths: Paths to the images for which you want to generate captions.
        :param generation_kwargs: Dictionary containing arguments for the generate method of the Hugging Face model.
                                  See [generate()](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate) in Hugging Face documentation.
        :param batch_size: Number of images to process at a time.
        :return: List of Documents. `Document.content` is the caption. `Document.meta["image_file_path"]` contains the path to the image file.
        """
        generation_kwargs = generation_kwargs or self.generation_kwargs
        batch_size = batch_size or self.batch_size

        if len(image_file_paths) == 0:
            raise ImageToTextError("ImageToText needs at least one file path to produce a caption.")

        if type(image_file_paths) is not list:
            raise ImageToTextError(
                "Expected List[str] for image_file_paths, got %s instead" % str(type(image_file_paths))
            )

        images_dataset = ListDataset(image_file_paths)

        captions: List[str] = []

        try:
            for captions_batch in tqdm(
                self.model(images_dataset, generate_kwargs=generation_kwargs, batch_size=batch_size),
                disable=not self.progress_bar,
                total=len(images_dataset),
                desc="Generating captions",
            ):
                captions.append("".join([el["generated_text"] for el in captions_batch]).strip())

        except Exception as exc:
            raise ImageToTextError(str(exc)) from exc

        result: List[Document] = []
        for caption, image_file_path in zip(captions, image_file_paths):
            document = Document(content=caption, content_type="text", meta={"image_path": image_file_path})
            result.append(document)

        return result
