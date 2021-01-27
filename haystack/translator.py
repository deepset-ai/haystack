import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from transformers import pipeline

from haystack import Document

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """
    Abstract class for Summarizer
    """

    outgoing_edges = 1

    @abstractmethod
    def translate(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        **kwargs
    ) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]:
        pass

    def run(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        **kwargs
    ):

        results: Dict = {
            "documents": [],
            **kwargs
        }

        if documents:
            results["documents"] = self.translate(documents=documents)
        if query:
            results["query"] = self.translate(query=query)

        return results, "output_1"


class TransformersTranslator(BaseTranslator):

    def __init__(
        self,
        # Refer https://huggingface.co/models?filter=translation for language code
        input_language_code: str,
        output_language_code: str,
        model_name_or_path: str = "t5-base",
        use_gpu: int = 0,
        tokenizer: Optional[str] = None,
        clean_up_tokenization_spaces: Optional[bool] = False,
    ):
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.input_language_code = input_language_code
        self.output_language_code = output_language_code
        # Naming convention of pipeline is translation_xx_to_yy
        pipeline_name = f'translation_{input_language_code}_to_{output_language_code}'
        self.model = pipeline(pipeline_name, model=model_name_or_path, tokenizer=tokenizer, device=use_gpu)

    def translate(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        **kwargs
    ) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]:
        if not query and not documents:
            raise AttributeError("Translator need query or documents to perform translation")

        if query and documents:
            raise AttributeError("Translator need either query or documents but not both")

        if documents and len(documents) == 0:
            logger.warning("Empty documents list is passed")
            return documents

        if isinstance(documents[0], Document):
            text_for_translator = [doc.text for doc in documents]
        elif isinstance(documents[0], str):
            text_for_translator = documents
        elif isinstance(documents[0], dict):
            if not isinstance(documents[0].get('text', None), str): # type: ignore
                raise AttributeError("Documents dictionary should have `text` key and it's value should be `str` type")
            text_for_translator = [doc.text for doc in documents]
        else:
            text_for_translator: List[str] = [query]

        translated_texts = self.model(
            text_for_translator,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )

        if query:
            return translated_texts[0]["translation_text"]

        if isinstance(documents[0], str):
            return [translated_text["translation_text"] for translated_text in translated_texts]

        for translated_text, doc in zip(translated_texts, documents):
            if isinstance(doc, Document):
                doc.text = translated_text["translation_text"]
            else:
                doc["text"] = translated_text["translation_text"]  # type: ignore

        return documents
