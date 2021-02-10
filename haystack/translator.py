import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from haystack import Document

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """
    Abstract class for a Translator component that translates either a query or a doc from language A to language B.
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
        """
        Translate the passed query or a list of documents from language A to B.
        """
        pass

    def run(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
        **kwargs
    ):
        """Method that gets executed when this class is used as a Node in a Haystack Pipeline"""

        results: Dict = {
            **kwargs
        }

        # This will cover input query stage
        if query:
            results["query"] = self.translate(query=query)
        # This will cover retriever and summarizer
        if documents:
            dict_key = dict_key or "text"
            results["documents"] = self.translate(documents=documents, dict_key=dict_key)

        if answers:
            dict_key = dict_key or "answer"
            if isinstance(answers, Mapping):
                # This will cover reader
                results["answers"] = self.translate(documents=answers["answers"], dict_key=dict_key)
            else:
                # This will cover generator
                results["answers"] = self.translate(documents=answers, dict_key=dict_key)

        return results, "output_1"


class TransformersTranslator(BaseTranslator):
    """
    Translator component based on Seq2Seq models from Huggingface's transformers library.
    Exemplary use cases:
    - Translate a query from Language A to B (e.g. if you only have good models + documents in language B)
    - Translate a document from Language A to B (e.g. if you want to return results in the native language of the user)

    We currently recommend using OPUS models (see __init__() for details)

    **Example:**

        ```python
        |    DOCS = [
        |        Document(text="Heinz von Foerster was an Austrian American scientist combining physics and philosophy,
        |                       and widely attributed as the originator of Second-order cybernetics.")
        |    ]
        |    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
        |    res = translator.translate(documents=DOCS, query=None)
        ```
    """
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name: Optional[str] = None
    ):
        """ Initialize the translator with a model that fits your targeted languages. While we support all seq2seq
        models from Hugging Face's model hub, we recommend using the OPUS models from Helsiniki NLP. They provide plenty
        of different models, usually one model per language pair and translation direction.
        They have a pretty standardized naming that should help you find the right model:
        - "Helsinki-NLP/opus-mt-en-de" => translating from English to German
        - "Helsinki-NLP/opus-mt-de-en" => translating from German to English
        - "Helsinki-NLP/opus-mt-fr-en" => translating from French to English
        - "Helsinki-NLP/opus-mt-hi-en"=> translating from Hindi to English
        ...

        They also have a few multilingual models that support multiple languages at once.

        :param model_name_or_path: Name of the seq2seq model that shall be used for translation.
                                   Can be a remote name from Huggingface's modelhub or a local path.
        :param tokenizer_name: Optional tokenizer name. If not supplied, `model_name_or_path` will also be used for the tokenizer.
        :param skip_special_tokens:
        """

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
        """
        Run the actual translation. You can supply a query or a list of documents. Whatever is supplied will be translated.
        """
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
            return translated_texts[0]
        elif documents:
            if isinstance(documents, list) and isinstance(documents[0], str):
                return [translated_text for translated_text in translated_texts]

            for translated_text, doc in zip(translated_texts, documents):
                if isinstance(doc, Document):
                    doc.text = translated_text
                else:
                    doc[dict_key] = translated_text  # type: ignore

            return documents

        raise AttributeError("Translator need query or documents to perform translation")
