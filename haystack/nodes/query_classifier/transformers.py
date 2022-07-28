import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple

from transformers import pipeline
from haystack.nodes.query_classifier.base import BaseQueryClassifier
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


class TransformersQueryClassifier(BaseQueryClassifier):
    """
    A node to classify an incoming query into one of two categories using a (small) BERT transformer model.
    Depending on the result, the query flows to a different branch in your pipeline and the further processing
    can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2`
    from this node.

    Example:
     ```python
        |{
        |pipe = Pipeline()
        |pipe.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
        |pipe.add_node(component=elastic_retriever, name="ElasticRetriever", inputs=["QueryClassifier.output_2"])
        |pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

        |# Keyword queries will use the ElasticRetriever
        |pipe.run("kubernetes aws")

        |# Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
        |pipe.run("How to manage kubernetes on aws")

     ```

    Models:

    Pass your own `Transformer` binary classification model from file/huggingface or use one of the following
    pretrained ones hosted on Huggingface:
    1) Keywords vs. Questions/Statements (Default)
       model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection"
       output_1 => question/statement
       output_2 => keyword query
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)


    2) Questions vs. Statements
    `model_name_or_path`="shahrukhx01/question-vs-statement-classifier"
     output_1 => question
     output_2 => statement
     [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)


    See also the [tutorial](https://haystack.deepset.ai/tutorials/pipelines) on pipelines.
    """

    def __init__(
        self,
        model_name_or_path: Union[Path, str] = "shahrukhx01/bert-mini-finetune-question-detection",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        task: str = "text-classification",
        labels: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'shahrukhx01/bert-mini-finetune-question-detection'.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU (if available).
        :param task: 'text-classification' or 'zero-shot-classification'
        :param labels: Only used for task 'zero-shot-classification'. List of string defining class labels, e.g.,
        ["positive", "negative"] otherwise None. Given a LABEL, the sequence fed to the model is "<cls> sequence to
        classify <sep> This example is LABEL . <sep>" and the model predicts whether that sequence is a contradiction
        or an entailment.
        :param batch_size: Number of Documents to be processed at a time.
        """
        super().__init__()
        devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        device = 0 if devices[0].type == "cuda" else -1
        self.model = pipeline(
            task=task, model=model_name_or_path, tokenizer=tokenizer, device=device, revision=model_version
        )
        if labels and task == "text-classification":
            logger.warning(
                f"Provided labels {labels} will be ignored for task text-classification. Set task to "
                f"zero-shot-classification to use labels."
            )
        if task == "zero-shot-classification":
            self.labels = labels
        elif task == "text-classification":
            self.labels = [k for k in self.model.model.config.label2id.keys()]

        self.task = task
        self.batch_size = batch_size

    def _get_edge_number(self, label):
        return self.labels.index(label) + 1

    def run(self, query: str):
        if self.task == "zero-shot-classification":
            prediction = self.model([query], candidate_labels=self.labels, truncation=True)
            label = prediction[0]["labels"][0]
        elif self.task == "text-classification":
            prediction = self.model([query], truncation=True)
            label = prediction[0]["label"]
        return {}, f"output_{self._get_edge_number(label)}"

    def run_batch(self, queries: List[str], batch_size: Optional[int] = None):  # type: ignore
        if batch_size is None:
            batch_size = self.batch_size
        if self.task == "zero-shot-classification":
            predictions = self.model(queries, candidate_labels=self.labels, truncation=True, batch_size=batch_size)
        elif self.task == "text-classification":
            prediction = self.model(queries, truncation=True, batch_size=batch_size)

        results: Dict[str, Dict[str, List]] = {
            f"output_{self._get_edge_number(label)}": {"queries": []} for label in self.labels
        }
        for query, prediction in zip(queries, predictions):
            if self.task == "zero-shot-classification":
                label = prediction[0]["labels"][0]
            elif self.task == "text-classification":
                label = prediction[0]["label"]
            results[f"output_{self._get_edge_number(label)}"]["queries"].append(query)

        return results, "split"
