import logging
from typing import List, Optional, Union, Dict
import itertools

import torch
from tqdm.auto import tqdm
from transformers import pipeline

from haystack.nodes.base import Document
from haystack.nodes.doc_language_classifier.base import BaseDocumentLanguageClassifier
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class TransformersDocumentLanguageClassifier(BaseDocumentLanguageClassifier):
    """
    Transformer-based model for classifying the document language using the Hugging Face's [transformers framework](https://github.com/huggingface/transformers).
    While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
    This node detects the language of Documents and adds the output to the Documents metadata.
    The meta field of the Document is a dictionary with the following format:
    ``'meta': {'name': '450_Baelor.txt', 'language': 'en'}``
    - Using the document language classifier, you can directly get predictions with the `predict()` method.
    - You can route the Documents to different branches depending on their language
      by setting the `route_by_language` parameter to True and specifying the `languages_to_route` parameter.
    **Usage example**
    ```python
    ...
    docs = [Document(content="The black dog runs across the meadow")]

    doclangclassifier = TransformersDocumentLanguageClassifier()
    results = doclangclassifier.predict(documents=docs)

    # print the predicted language
    print(results[0].to_dict()["meta"]["language"]

    **Usage example for routing**
    ```python
    ...
    docs = [Document(content="My name is Ryan and I live in London"),
            Document(content="Mi chiamo Matteo e vivo a Roma")]

    doclangclassifier = TransformersDocumentLanguageClassifier(
        route_by_language = True,
        languages_to_route = ['en','it','es']
        )
    for doc in docs:
        doclangclassifier.run(doc)
    ```
    """

    def __init__(
        self,
        route_by_language: bool = True,
        languages_to_route: Optional[List[str]] = None,
        labels_to_languages_mapping: Optional[Dict[str, str]] = None,
        model_name_or_path: str = "papluca/xlm-roberta-base-language-detection",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Load a language detection model from Transformers.
        For a full list of available models, see [Hugging Face models](https://huggingface.co/models).
        For language detection models, see [Language Detection models](https://huggingface.co/models?search=language%20detection) on Hugging Face.

        :param route_by_language: Sends Documents to a different output edge depending on their language.
        :param languages_to_route: A list of languages, each corresponding to a different output edge (for the list of supported languages, see the model card of the chosen model).
        :param labels_to_languages_mapping: Some Transformers models return generic labels instead of language names. In this case, you can provide a mapping indicating a language for each label. For example: {"LABEL_1": "ar", "LABEL_2": "bg", ...}.

        :param model_name_or_path: Directory of a saved model or the name of a public model, for example 'papluca/xlm-roberta-base-language-detection'.
        See [Hugging Face models](https://huggingface.co/models) for a full list of available models.
        :param model_version: The version of the model to use from the Hugging Face model hub. Can be a tag name, a branch name, or a commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model).
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param progress_bar: Whether to show a progress bar while processing.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                               If set to `True`, the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) is used.
                               For more information, see [Hugging Face documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained).
        :param devices: List of torch devices (for example, cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects or strings is supported (for example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False`, the devices
                        parameter is not used and a single cpu device is used for inference.

        """
        super().__init__(route_by_language=route_by_language, languages_to_route=languages_to_route)

        resolved_devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(resolved_devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        if tokenizer is None:
            tokenizer = model_name_or_path

        self.model = pipeline(
            task="text-classification",
            model=model_name_or_path,
            tokenizer=tokenizer,
            device=resolved_devices[0],
            revision=model_version,
            top_k=1,
            use_auth_token=use_auth_token,
        )
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.labels_to_languages_mapping = labels_to_languages_mapping or {}

    def predict(self, documents: List[Document], batch_size: Optional[int] = None) -> List[Document]:
        """
        Detect the language of Documents and add the output to the Documents metadata.
        :param documents: A list of Documents whose language you want to detect.
        :param batch_size: The number of Documents to classify at a time.
        :return: A list of Documents, where Document.meta["language"] contains the predicted language.
        """
        if len(documents) == 0:
            raise ValueError(
                "TransformersDocumentLanguageClassifier needs at least one document to predict the language."
            )
        if batch_size is None:
            batch_size = self.batch_size

        texts = [doc.content for doc in documents]
        batches = self._get_batches(texts, batch_size=batch_size)
        predictions = []
        pb = tqdm(total=len(texts), disable=not self.progress_bar, desc="Predicting the language of documents")
        for batch in batches:
            batched_prediction = self.model(batch, top_k=1, truncation=True)
            predictions.extend(batched_prediction)
            pb.update(len(batch))
        pb.close()
        for prediction, doc in zip(predictions, documents):
            label = prediction[0]["label"]
            # replace the label with the language, if present in the mapping
            language = self.labels_to_languages_mapping.get(label, label)
            doc.meta["language"] = language
        return documents

    def predict_batch(self, documents: List[List[Document]], batch_size: Optional[int] = None) -> List[List[Document]]:
        """
        Detect the Document's language and add the output to the Document's meta data.
        :param documents: A list of lists of Documents whose language you want to detect.
        :return: A list of lists of Documents where Document.meta["language"] contains the predicted language.
        """
        if len(documents) == 0 or all(len(docs_list) == 0 for docs_list in documents):
            raise ValueError(
                "TransformersDocumentLanguageClassifier needs at least one document to predict the language."
            )
        if batch_size is None:
            batch_size = self.batch_size

        flattened_documents = list(itertools.chain.from_iterable(documents))
        docs_with_preds = self.predict(flattened_documents, batch_size=batch_size)

        # Group documents together
        grouped_documents = []
        for docs_list in documents:
            grouped_documents.append(docs_with_preds[: len(docs_list)])
            docs_with_preds = docs_with_preds[len(docs_list) :]

        return grouped_documents

    def _get_batches(self, items, batch_size):
        if batch_size is None:
            yield items
            return
        for index in range(0, len(items), batch_size):
            yield items[index : index + batch_size]
