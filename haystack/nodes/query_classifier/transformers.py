import logging
from pathlib import Path
from typing import Union, List, Optional

from transformers import pipeline
from haystack.nodes.query_classifier.base import BaseQueryClassifier
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


class TransformersQueryClassifier(BaseQueryClassifier):

    outgoing_edges: int = 10

    """
    A node to classify an incoming query into categories using a transformer model.
    Depending on the result, the query flows to a different branch in your pipeline and the further processing
    can be customized. You can define this by connecting the further pipeline to `output_1`, `output_2`, ..., `output_n`
    from this node.
    This node also supports zero-shot-classification.

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

    Pass your own `Transformer` classification/zero-shot-classification model from file/huggingface or use one of the following
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
        labels: Optional[List[str]] = ["LABEL_1", "LABEL_0"],
        batch_size: Optional[int] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model, for example 'shahrukhx01/bert-mini-finetune-question-detection'.
        See [Hugging Face models](https://huggingface.co/models) for a full list of available models.
        :param model_version: The version of the model to use from the Hugging Face model hub. This can be a tag name, a branch name, or a commit hash.
        :param tokenizer: The name of the tokenizer (usually the same as model).
        :param use_gpu: Whether to use GPU (if available).
        :param task: Specifies the type of classification. Possible values: 'text-classification' or 'zero-shot-classification'.
        :param labels: If the task is 'text-classification' and an ordered list of labels is provided, the first label corresponds to output_1,
        the second label to output_2, and so on. The labels must match the model labels; only the order can differ. Otherwise, model labels are considered.
        If the task is 'zero-shot-classification', these are the candidate labels.
        :param batch_size: Number of queries to be processed at a time.
        """
        super().__init__()
        devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        device = 0 if devices[0].type == "cuda" else -1

        self.model = pipeline(
            task=task, model=model_name_or_path, tokenizer=tokenizer, device=device, revision=model_version
        )

        self.labels = labels
        if task == "zero-shot-classification":
            if labels is None:
                raise ValueError("Candidate labels must be provided for task zero-shot-classification")
        elif task == "text-classification":
            labels_from_model = [label for label in self.model.model.config.id2label.values()]
            if labels is None or set(labels) != set(labels_from_model):
                self.labels = labels_from_model
                logger.warning(
                    f"The labels are not provided or do not match the model labels. Then the model labels are used.\n"
                    f"Provided labels: {labels}\n"
                    f"Model labels: {labels_from_model}"
                )
        self.task = task
        self.batch_size = batch_size

    def _get_edge_number(self, label):
        return self.labels.index(label) + 1

    def run(self, query: str):  # type: ignore
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
            predictions = self.model(queries, truncation=True, batch_size=batch_size)

        results = {f"output_{self._get_edge_number(label)}": {"queries": []} for label in self.labels}  # type: ignore
        for query, prediction in zip(queries, predictions):
            if self.task == "zero-shot-classification":
                label = prediction["labels"][0]
            elif self.task == "text-classification":
                label = prediction["label"]
            results[f"output_{self._get_edge_number(label)}"]["queries"].append(query)

        return results, "split"
