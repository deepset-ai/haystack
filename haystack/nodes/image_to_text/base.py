from typing import List, Optional

from abc import abstractmethod

from haystack.schema import Document
from haystack.nodes.base import BaseComponent


class BaseImageToText(BaseComponent):
    """
    Abstract class for ImageToText
    """

    outgoing_edges = 1

    @abstractmethod
    def generate_captions(
        self, image_file_paths: List[str], generate_kwargs: Optional[dict] = None, batch_size: Optional[int] = None
    ) -> List[Document]:
        """
        Abstract method for generating captions.

        :param image_file_paths: Paths of the images
        :param generate_kwargs: Dictionary containing arguments for the generate method of the Hugging Face model.
                                See https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        :param batch_size: Number of images to process at a time.
        :return: List of Documents. Document.content is the caption. Document.meta["image_file_path"] contains the image file path.
        """
        pass

    def run(self, file_paths: Optional[List[str]] = None, documents: Optional[List[Document]] = None):  # type: ignore

        if file_paths is None and documents is None:
            raise ValueError("You must either specify documents or image file_paths to process.")
        if file_paths is not None and documents is not None:
            raise ValueError(
                "You specified both documents and image_file_paths. You need to specify only one of the two parameters."
            )
        if file_paths is not None:
            image_file_paths = file_paths
        if documents is not None:
            if any((doc.content_type != "image" for doc in documents)):
                raise ValueError("The ImageToText node only supports image documents.")
            image_file_paths = [doc.content for doc in documents]

        results: dict = {}
        results["documents"] = self.generate_captions(image_file_paths=image_file_paths)

        return results, "output_1"

    def run_batch(
        self, file_paths: Optional[List[str]] = None, documents: Optional[List[Document]] = None
    ):  # type: ignore

        return self.run(file_paths=file_paths, documents=documents)
