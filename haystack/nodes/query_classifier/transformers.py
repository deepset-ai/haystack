import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

import torch
from transformers import pipeline
from tqdm.auto import tqdm

from haystack.nodes.query_classifier.base import BaseQueryClassifier
from haystack.modeling.utils import initialize_device_settings
from haystack.utils.torch_utils import ListDataset

logger = logging.getLogger(__name__)

DEFAULT_LABELS = ["LABEL_1", "LABEL_0"]


class TransformersQueryClassifier(BaseQueryClassifier):
    """
    A node to classify an incoming query into categories using a transformer model.
    Depending on the result, the query flows to a different branch in your pipeline and the further processing
    can be customized. You can define this by connecting the further pipeline to `output_1`, `output_2`, ..., `output_n`
    from this node.
    This node also supports zero-shot-classification.

    Example:
    ```python
    {
    pipe = Pipeline()
    pipe.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipe.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["QueryClassifier.output_2"])
    pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

    # Keyword queries will use the BM25Retriever
    pipe.run("kubernetes aws")

    # Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
    pipe.run("How to manage kubernetes on aws")

    ```

    Models:

    Pass your own `Transformer` classification/zero-shot-classification model from file/huggingface or use one of the following
    pretrained ones hosted on Huggingface:
    1) Keywords vs. Questions/Statements (Default)
       model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection"
       output_1 => question/statement
       output_2 => keyword query
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_2022/readme.txt)


    2) Questions vs. Statements
    `model_name_or_path`="shahrukhx01/question-vs-statement-classifier"
     output_1 => question
     output_2 => statement
     [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements_2022/readme.txt)


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
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model, for example 'shahrukhx01/bert-mini-finetune-question-detection'.
        See [Hugging Face models](https://huggingface.co/models) for a full list of available models.
        :param model_version: The version of the model to use from the Hugging Face model hub. This can be a tag name, a branch name, or a commit hash.
        :param tokenizer: The name of the tokenizer (usually the same as model).
        :param use_gpu: Whether to use GPU (if available).
        :param task: Specifies the type of classification. Possible values: 'text-classification' or 'zero-shot-classification'.
        :param labels: If the task is 'text-classification' and an ordered list of labels is provided, the first label corresponds to output_1,
        the second label to output_2, and so on. The labels must match the model labels; only the order can differ.
        If the task is 'zero-shot-classification', these are the candidate labels.
        :param batch_size: The number of queries to be processed at a time.
        :param progress_bar: Whether to show a progress bar.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        if labels is None:
            labels = DEFAULT_LABELS
        super().__init__()
        resolved_devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(resolved_devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )

        self.model = pipeline(
            task=task,
            model=model_name_or_path,
            tokenizer=tokenizer,
            device=resolved_devices[0],
            revision=model_version,
            use_auth_token=use_auth_token,
        )

        self.labels = labels
        if task == "text-classification":
            labels_from_model = [label for label in self.model.model.config.id2label.values()]
            if set(labels) != set(labels_from_model):
                raise ValueError(
                    f"For text-classification, the provided labels must match the model labels; only the order can differ.\n"
                    f"Provided labels: {labels}\n"
                    f"Model labels: {labels_from_model}"
                )
        if task not in ["text-classification", "zero-shot-classification"]:
            raise ValueError(
                f"Task not supported: {task}.\n"
                f"Possible task values are: 'text-classification' or 'zero-shot-classification'"
            )
        self.task = task
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        labels = component_params.get("labels", DEFAULT_LABELS)
        if labels is None or len(labels) == 0:
            raise ValueError("The labels must be provided")
        return len(labels)

    def _get_edge_number_from_label(self, label):
        return self.labels.index(label) + 1

    def run(self, query: str):  # type: ignore
        if self.task == "zero-shot-classification":
            prediction = self.model([query], candidate_labels=self.labels, truncation=True)
            label = prediction[0]["labels"][0]
        elif self.task == "text-classification":
            prediction = self.model([query], truncation=True)
            label = prediction[0]["label"]
        return {}, f"output_{self._get_edge_number_from_label(label)}"

    def run_batch(self, queries: List[str], batch_size: Optional[int] = None):  # type: ignore
        # HF pb hack https://discuss.huggingface.co/t/progress-bar-for-hf-pipelines/20498/2
        queries_dataset = ListDataset(queries)
        if batch_size is None:
            batch_size = self.batch_size
        all_predictions = []
        if self.task == "zero-shot-classification":
            for predictions in tqdm(
                self.model(queries_dataset, candidate_labels=self.labels, truncation=True, batch_size=batch_size),
                disable=not self.progress_bar,
                desc="Classifying queries",
            ):
                all_predictions.extend([predictions])
        elif self.task == "text-classification":
            for predictions in tqdm(
                self.model(queries_dataset, truncation=True, batch_size=batch_size),
                disable=not self.progress_bar,
                desc="Classifying queries",
            ):
                all_predictions.extend([predictions])
        results = {f"output_{self._get_edge_number_from_label(label)}": {"queries": []} for label in self.labels}  # type: ignore
        for query, prediction in zip(queries, all_predictions):
            if self.task == "zero-shot-classification":
                label = prediction["labels"][0]
            elif self.task == "text-classification":
                label = prediction["label"]
            results[f"output_{self._get_edge_number_from_label(label)}"]["queries"].append(query)

        return results, "split"
