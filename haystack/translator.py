import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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
        dict_key: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]:
        pass

    def run(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
        **kwargs
    ):

        results: Dict = {
            "documents": [],
            **kwargs
        }

        if documents:
            results["documents"] = self.translate(documents=documents, dict_key=dict_key)
        if query:
            results["query"] = self.translate(query=query)

        return results, "output_1"


class TransformersTranslator(BaseTranslator):

    def __init__(
        self,
        # Refer https://huggingface.co/models?filter=translation for language code
        # Opus models are preferred https://huggingface.co/Helsinki-NLP
        # Currently do not support multilingual model
        model_name_or_path: str,
        tokenizer_name: Optional[str] = None,
        skip_special_tokens: Optional[bool] = True,
    ):
        self.skip_special_tokens = skip_special_tokens
        tokenizer_name = tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def translate(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]:
        if not query and not documents:
            raise AttributeError("Translator need query or documents to perform translation")

        if query and documents:
            raise AttributeError("Translator need either query or documents but not both")

        if documents and len(documents) == 0:
            logger.warning("Empty documents list is passed")
            return documents

        dict_key = dict_key or "text"

        if isinstance(documents, list):
            if isinstance(documents[0], Document):
                text_for_translator = [doc.text for doc in documents]   # type: ignore
            elif isinstance(documents[0], str):
                text_for_translator = documents   # type: ignore
            else:
                if not isinstance(documents[0].get(dict_key, None), str):    # type: ignore
                    raise AttributeError(f"Dictionary should have {dict_key} key and it's value should be `str` type")
                text_for_translator = [doc[dict_key] for doc in documents]    # type: ignore
        else:
            text_for_translator: List[str] = [query]     # type: ignore

        batch = self.tokenizer.prepare_seq2seq_batch(src_texts=text_for_translator, return_tensors="pt")
        generated_output = self.model.generate(**batch)
        translated_texts = self.tokenizer.batch_decode(generated_output, skip_special_tokens=self.skip_special_tokens)

        if query:
            return translated_texts[0]["translation_text"]
        elif documents:
            if isinstance(documents, list) and isinstance(documents[0], str):
                return [translated_text["translation_text"] for translated_text in translated_texts]

            for translated_text, doc in zip(translated_texts, documents):
                if isinstance(doc, Document):
                    doc.text = translated_text["translation_text"]
                else:
                    doc[dict_key] = translated_text["translation_text"]  # type: ignore

            return documents

        raise AttributeError("Translator need query or documents to perform translation")
