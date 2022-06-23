import logging
from typing import Any, Dict, List, Optional, Union

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from haystack.errors import HaystackError
from haystack.schema import Document, Answer
from haystack.nodes.translator.base import BaseTranslator
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


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
        |        Document(content="Heinz von Foerster was an Austrian American scientist combining physics and philosophy,
        |                       and widely attributed as the originator of Second-order cybernetics.")
        |    ]
        |    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
        |    res = translator.translate(documents=DOCS, query=None)
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        clean_up_tokenization_spaces: Optional[bool] = True,
        use_gpu: bool = True,
    ):
        """Initialize the translator with a model that fits your targeted languages. While we support all seq2seq
        models from Hugging Face's model hub, we recommend using the OPUS models from Helsinki NLP. They provide plenty
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
        :param tokenizer_name: Optional tokenizer name. If not supplied, `model_name_or_path` will also be used for the
                               tokenizer.
        :param max_seq_len: The maximum sentence length the model accepts. (Optional)
        :param clean_up_tokenization_spaces: Whether or not to clean up the tokenization spaces. (default True)
        :param use_gpu: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        self.max_seq_len = max_seq_len
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        tokenizer_name = tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(str(self.devices[0]))

    def translate(
        self,
        results: List[Dict[str, Any]] = None,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
    ) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]:
        """
        Run the actual translation. You can supply a query or a list of documents. Whatever is supplied will be translated.
        :param results: Generated QA pairs to translate
        :param query: The query string to translate
        :param documents: The documents to translate
        :param dict_key: If you pass a dictionary in `documents`, you can specify here the field which shall be translated.
        """
        queries_for_translator = None
        answers_for_translator = None
        if results is not None:
            queries_for_translator = [result["query"] for result in results]
            answers_for_translator = [result["answers"][0].answer for result in results]
        if not query and not documents and results is None:
            raise AttributeError("Translator needs query or documents to perform translation.")

        if query and documents:
            raise AttributeError("Translator needs either query or documents but not both.")

        if documents and len(documents) == 0:
            logger.warning("Empty documents list is passed")
            return documents

        dict_key = dict_key or "content"

        if queries_for_translator is not None and answers_for_translator is not None:
            text_for_translator = queries_for_translator + answers_for_translator

        elif isinstance(documents, list):
            if isinstance(documents[0], Document):
                text_for_translator = [doc.content for doc in documents]  # type: ignore
            elif isinstance(documents[0], Answer):
                text_for_translator = [answer.answer for answer in documents]  # type: ignore
            elif isinstance(documents[0], str):
                text_for_translator = documents  # type: ignore
            else:
                if not isinstance(documents[0].get(dict_key, None), str):  # type: ignore
                    raise AttributeError(f"Dictionary should have {dict_key} key and it's value should be `str` type")
                text_for_translator = [doc[dict_key] for doc in documents]  # type: ignore
        else:
            text_for_translator: List[str] = [query]  # type: ignore

        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=text_for_translator, return_tensors="pt", max_length=self.max_seq_len
        ).to(self.devices[0])
        generated_output = self.model.generate(**batch)
        translated_texts = self.tokenizer.batch_decode(
            generated_output, skip_special_tokens=True, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )

        if queries_for_translator is not None and answers_for_translator is not None:
            return translated_texts
        elif query:
            return translated_texts[0]
        elif documents:
            if isinstance(documents, list) and isinstance(documents[0], str):
                return [translated_text for translated_text in translated_texts]

            for translated_text, doc in zip(translated_texts, documents):
                if isinstance(doc, Document):
                    doc.content = translated_text
                elif isinstance(doc, Answer):
                    doc.answer = translated_text
                else:
                    doc[dict_key] = translated_text  # type: ignore

            return documents

        raise AttributeError("Translator need query or documents to perform translation")

    def translate_batch(
        self,
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[Answer], List[List[Document]], List[List[Answer]]]] = None,
        batch_size: Optional[int] = None,
    ) -> Union[str, List[str], List[Document], List[Answer], List[List[Document]], List[List[Answer]]]:
        """
        Run the actual translation. You can supply a single query, a list of queries or a list (of lists) of documents.

        :param queries: Single query or list of queries.
        :param documents: List of documents or list of lists of documets.
        :param batch_size: Not applicable.
        """
        # TODO: This method currently just calls the translate method multiple times, so there is room for improvement.

        if queries and documents:
            raise AttributeError("Translator needs either query or documents but not both.")

        if not queries and not documents:
            raise AttributeError("Translator needs query or documents to perform translation.")

        # Translate queries
        if queries:
            translated = []
            for query in queries:
                cur_translation = self.run(query=query)
                translated.append(cur_translation)

        # Translate docs / answers
        elif documents:
            # Single list of documents / answers
            if not isinstance(documents[0], list):
                translated = self.translate(documents=documents)  # type: ignore
            # Multiple lists of document / answer lists
            else:
                translated = []
                for cur_list in documents:
                    if not isinstance(cur_list, list):
                        raise HaystackError(
                            f"cur_list was of type {type(cur_list)}, but expected a list of " f"Documents / Answers."
                        )
                    cur_translation = self.translate(documents=cur_list)
                    translated.append(cur_translation)

        return translated
