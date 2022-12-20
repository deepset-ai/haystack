import logging
from pathlib import Path
from typing import Union, Any, List, Optional, Iterator, Dict
import pickle
import urllib

from tqdm.auto import tqdm
from sklearn.ensemble._gb_losses import BinomialDeviance
from sklearn.ensemble._gb import GradientBoostingClassifier

from haystack.nodes.query_classifier.base import BaseQueryClassifier


logger = logging.getLogger(__name__)


class SklearnQueryClassifier(BaseQueryClassifier):
    """
    A node to classify an incoming query into one of two categories using a lightweight sklearn model. Depending on the result, the query flows to a different branch in your pipeline
    and the further processing can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2` from this node.

    Example:

    ```python
    pipe = Pipeline()
    pipe.add_node(component=SklearnQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipe.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["QueryClassifier.output_2"])
    pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

    # Keyword queries will use the BM25Retriever
    pipe.run("kubernetes aws")

    # Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
    pipe.run("How to manage kubernetes on aws")

    ```

    Models:

    Pass your own `Sklearn` binary classification model or use one of the following pretrained ones:
    1) Keywords vs. Questions/Statements (Default)
       query_classifier can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle)
       query_vectorizer can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle)
       output_1 => question/statement
       output_2 => keyword query
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)


    2) Questions vs. Statements
       query_classifier can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/model.pickle)
       query_vectorizer can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/vectorizer.pickle)
       output_1 => question
       output_2 => statement
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)

    See also the [tutorial](https://haystack.deepset.ai/tutorials/pipelines) on pipelines.

    """

    def __init__(
        self,
        model_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle",
        vectorizer_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle",
        batch_size: Optional[int] = None,
        progress_bar: bool = True,
    ):
        """
        :param model_name_or_path: Gradient boosting based binary classifier to classify between keyword vs statement/question
        queries or statement vs question queries.
        :param vectorizer_name_or_path: A ngram based Tfidf vectorizer for extracting features from query.
        :param batch_size: Number of queries to process at a time.
        :param progress_bar: Whether to show a progress bar.
        """
        if ((not isinstance(model_name_or_path, Path)) and (not isinstance(model_name_or_path, str))) or (
            (not isinstance(vectorizer_name_or_path, Path)) and (not isinstance(vectorizer_name_or_path, str))
        ):
            raise TypeError("model_name_or_path and vectorizer_name_or_path must either be of type Path or str")

        super().__init__()

        if isinstance(model_name_or_path, Path):
            file_url = urllib.request.pathname2url(r"{}".format(model_name_or_path))
            model_name_or_path = f"file:{file_url}"

        if isinstance(vectorizer_name_or_path, Path):
            file_url = urllib.request.pathname2url(r"{}".format(vectorizer_name_or_path))
            vectorizer_name_or_path = f"file:{file_url}"

        self.model = pickle.load(urllib.request.urlopen(model_name_or_path))
        # MONKEY PATCH for scikit-learn>1.0.2
        # see https://github.com/deepset-ai/haystack/issues/2904
        if isinstance(self.model, GradientBoostingClassifier) and not hasattr(self.model, "_loss"):
            self.model._loss = BinomialDeviance(2)

        self.vectorizer = pickle.load(urllib.request.urlopen(vectorizer_name_or_path))
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def run(self, query):
        query_vector = self.vectorizer.transform([query])

        is_question: bool = self.model.predict(query_vector)[0]
        if is_question:
            return {}, "output_1"
        else:
            return {}, "output_2"

    def run_batch(self, queries: List[str], batch_size: Optional[int] = None):  # type: ignore
        if batch_size is None:
            batch_size = self.batch_size

        split: Dict[str, Dict[str, List]] = {"output_1": {"queries": []}, "output_2": {"queries": []}}

        batches = self._get_batches(queries=queries, batch_size=batch_size)
        predictions = []
        pb = tqdm(batches, total=len(queries), disable=self.progress_bar, desc="Classifying queries")
        for batch in batches:
            query_vectors = self.vectorizer.transform(batch)
            predictions.extend(self.model.predict(query_vectors))
            pb.update(len(batch))
        pb.close()
        for query, pred in zip(queries, predictions):
            if pred:
                split["output_1"]["queries"].append(query)
            else:
                split["output_2"]["queries"].append(query)

        return split, "split"

    @staticmethod
    def _get_batches(queries: List[str], batch_size: Optional[int]) -> Iterator[List[str]]:
        if batch_size is None:
            yield queries
            return
        else:
            for index in range(0, len(queries), batch_size):
                yield queries[index : index + batch_size]
