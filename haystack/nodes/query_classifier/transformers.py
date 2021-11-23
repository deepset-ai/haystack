import logging
from pathlib import Path
from typing import Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from haystack.nodes.query_classifier import BaseQueryClassifier
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
        use_gpu: bool = True,
    ):
        """
        :param model_name_or_path: Transformer based fine tuned mini bert model for query classification
        :param use_gpu: Whether to use GPU (if available).
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(model_name_or_path=model_name_or_path)
        self.devices, _ = initialize_device_settings(use_cuda=use_gpu)
        device = 0 if self.devices[0].type == "cuda" else -1

        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.query_classification_pipeline = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, device=device
        )

    def run(self, query):
        is_question: bool = (
            self.query_classification_pipeline(query)[0]["label"] == "LABEL_1"
        )

        if is_question:
            return {}, "output_1"
        else:
            return {}, "output_2"
