import logging
from typing import List, Optional

from transformers import pipeline

from haystack import Document
from haystack.document_classifier.base import BaseDocumentClassifier

logger = logging.getLogger(__name__)


class TransformersDocumentClassifier(BaseDocumentClassifier):
    """
    Transformer based model for document classification using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
    This node classifies documents and adds the output from the classification step to the document's meta data.
    The meta field of the document is a dictionary with the following format:
    ``'meta': {'name': '450_Baelor.txt', 'classification': {'label': 'neutral', 'probability' = 0.9997646, ...} }``

    With this document_classifier, you can directly get predictions via predict()
    
     **Usage example:**
     ```python
    |    ...
    |    retriever = ElasticsearchRetriever(document_store=document_store)
    |    document_classifier = TransformersDocumentClassifier(model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion")
    |    p = Pipeline()
    |    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    |    p.add_node(component=document_classifier, name="Classifier", inputs=["Retriever"])
    |    res = p.run(
    |        query="Who is the father of Arya Stark?",
    |        params={"Retriever": {"top_k": 10}}
    |    )
    |    
    |    # print the classification results
    |    print_documents(res, max_text_len=100, print_meta=True)
    |    # or access the predicted class label directly
    |    res["documents"][0].to_dict()["meta"]["classification"]["label"]
     ```
    """

    def __init__(
        self,
        model_name_or_path: str = "bhadresh-savani/distilbert-base-uncased-emotion",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: int = 0,
        return_all_scores: bool = False,
        task: str = 'text-classification',
        labels: Optional[List[str]] = None
    ):
        """
        Load a text classification model from Transformers.
        Available models for the task of text-classification include:
        - ``'bhadresh-savani/distilbert-base-uncased-emotion'``
        - ``'Hate-speech-CNERG/dehatebert-mono-english'``

        Available models for the task of zero-shot-classification include:
        - ``'valhalla/distilbart-mnli-12-3'``
        - ``'cross-encoder/nli-distilroberta-base'``

        See https://huggingface.co/models for full list of available models.
        Filter for text classification models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
        Filter for zero-shot classification models (NLI): https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads&search=nli

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bhadresh-savani/distilbert-base-uncased-emotion'.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
        :param return_all_scores:  Whether to return all prediction scores or just the one of the predicted class. Only used for task 'text-classification'.
        :param task: 'text-classification' or 'zero-shot-classification'
        :param labels: Only used for task 'zero-shot-classification'. List of string defining class labels, e.g.,
        ["positive", "negative"] otherwise None. Given a LABEL, the sequence fed to the model is "<cls> sequence to
        classify <sep> This example is LABEL . <sep>" and the model predicts whether that sequence is a contradiction
        or an entailment.

        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version, tokenizer=tokenizer,
            use_gpu=use_gpu, return_all_scores=return_all_scores, labels=labels, task=task
        )
        if labels and task == 'text-classification':
            logger.warning(f'Provided labels {labels} will be ignored for task text-classification. Set task to '
                           f'zero-shot-classification to use labels.')

        if tokenizer is None:
            tokenizer = model_name_or_path
        if task == 'zero-shot-classification':
            self.model = pipeline(task=task, model=model_name_or_path, tokenizer=tokenizer, device=use_gpu, revision=model_version)
        elif task == 'text-classification':
            self.model = pipeline(task=task, model=model_name_or_path, tokenizer=tokenizer, device=use_gpu, revision=model_version, return_all_scores=return_all_scores)
        self.return_all_scores = return_all_scores
        self.labels = labels
        self.task = task

    def predict(self, documents: List[Document]) -> List[Document]:
        """
        Returns documents containing classification result in meta field

        :param documents: List of Document to classify
        :return: List of Document enriched with meta information

        """
        texts = [doc.text for doc in documents]
        if self.task == 'zero-shot-classification':
            predictions = self.model(texts, candidate_labels=self.labels, truncation=True)
        elif self.task == 'text-classification':
            predictions = self.model(texts, return_all_scores=self.return_all_scores, truncation=True)

        classified_docs: List[Document] = []

        for prediction, doc in zip(predictions, documents):
            cur_doc = doc
            cur_doc.meta["classification"] = prediction
            if self.task == 'zero-shot-classification':
                cur_doc.meta["classification"]["label"] = cur_doc.meta["classification"]["labels"][0]
            classified_docs.append(cur_doc)

        return classified_docs
