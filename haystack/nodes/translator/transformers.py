import logging
from typing import Any, Dict, List, Optional, Union

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from haystack.schema import Document, Answer
from haystack.nodes.translator import BaseTranslator
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
        tokenizer_name: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        clean_up_tokenization_spaces: Optional[bool] = True,
        use_gpu: bool = True,
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
        :param tokenizer_name: Optional tokenizer name. If not supplied, `model_name_or_path` will also be used for the
                               tokenizer.
        :param max_seq_len: The maximum sentence length the model accepts. (Optional)
        :param clean_up_tokenization_spaces: Whether or not to clean up the tokenization spaces. (default True)
        :param use_gpu: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, tokenizer_name=tokenizer_name, max_seq_len=max_seq_len,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        self.max_seq_len = max_seq_len
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        tokenizer_name = tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(str(self.devices[0]))

    def translate(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
    ) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]:
        """
        Run the actual translation. You can supply a query or a list of documents. Whatever is supplied will be translated.
        :param query: The query string to translate
        :param documents: The documents to translate
        :param dict_key: If you pass a dictionary in `documents`, you can specify here the field which shall be translated.
        """
        if not query and not documents:
            raise AttributeError("Translator need query or documents to perform translation")

        if query and documents:
            raise AttributeError("Translator need either query or documents but not both")

        if documents and len(documents) == 0:
            logger.warning("Empty documents list is passed")
            return documents

        dict_key = dict_key or "content"

        if isinstance(documents, list):
            if isinstance(documents[0], Document):
                text_for_translator = [doc.content for doc in documents]   # type: ignore
            elif isinstance(documents[0], Answer):
                text_for_translator = [answer.answer for answer in documents] # type: ignore
            elif isinstance(documents[0], str):
                text_for_translator = documents   # type: ignore
            else:
                if not isinstance(documents[0].get(dict_key, None), str):    # type: ignore
                    raise AttributeError(f"Dictionary should have {dict_key} key and it's value should be `str` type")
                text_for_translator = [doc[dict_key] for doc in documents]    # type: ignore
        else:
            text_for_translator: List[str] = [query]     # type: ignore

        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=text_for_translator,
            return_tensors="pt",
            max_length=self.max_seq_len
        ).to(self.devices[0])
        generated_output = self.model.generate(**batch)
        translated_texts = self.tokenizer.batch_decode(
            generated_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )

        if query:
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
