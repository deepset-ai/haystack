import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.infer import Inferencer
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

from haystack import Document
from haystack.classifier.base import BaseClassifier

logger = logging.getLogger(__name__)


class FARMClassifier(BaseClassifier):
    """
    This node classifies documents and adds the output from the classification step to the document's meta data.
    The meta field of the document is a dictionary with the following format:
    'meta': {'name': '450_Baelor.txt', 'classification': {'label': 'neutral', 'probability' = 0.9997646, ...} }

    |  With a FARMClassifier, you can:
     - directly get predictions via predict()
     - fine-tune the model on text classification training data via train()

    Usage example:
    ...
    retriever = ElasticsearchRetriever(document_store=document_store)
    classifier = FARMClassifier(model_name_or_path="deepset/bert-base-german-cased-sentiment-Germeval17")
    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=classifier, name="Classifier", inputs=["ESRetriever"])

    res = p_extractive.run(
        query="Who is the father of Arya Stark?",
        top_k_retriever=10,
        top_k_reader=5
    )

    print(res["documents"][0].to_dict()["meta"]["classification"]["label"])
    # Note that print_documents() does not output the content of the classification field in the meta data
    # document_dicts = [doc.to_dict() for doc in res["documents"]]
    # res["documents"] = document_dicts
    # print_documents(res, max_text_len=100)
    """

    def __init__(
            self,
            model_name_or_path: Union[str, Path],
            model_version: Optional[str] = None,
            batch_size: int = 50,
            use_gpu: bool = True,
            top_k: int = 10,
            num_processes: Optional[int] = None,
            max_seq_len: int = 256,
            progress_bar: bool = True
    ):

        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'deepset/bert-base-german-cased-sentiment-Germeval17'.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param batch_size: Number of samples the model receives in one batch for inference.
                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                           to a value so only a single batch is used.
        :param use_gpu: Whether to use GPU (if available)
        :param top_k: The maximum number of documents to return
        :param num_processes: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer determine optimum number. If you
                              want to debug the Language Model, you might need to disable multiprocessing!
        :param max_seq_len: Max sequence length of one input text for the model
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version,
            batch_size=batch_size, use_gpu=use_gpu, top_k=top_k,
            num_processes=num_processes, max_seq_len=max_seq_len, progress_bar=progress_bar,
        )

        self.top_k = top_k

        self.inferencer = Inferencer.load(model_name_or_path, batch_size=batch_size, gpu=use_gpu,
                                          task_type="text_classification", max_seq_len=max_seq_len,
                                          num_processes=num_processes, revision=model_version,
                                          disable_tqdm=not progress_bar,
                                          strict=False)

        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar

    def train(
            self,
            data_dir: str,
            train_filename: str,
            label_list: List[str],
            delimiter: str,
            metric: str,
            dev_filename: Optional[str] = None,
            test_filename: Optional[str] = None,
            use_gpu: Optional[bool] = None,
            batch_size: int = 10,
            n_epochs: int = 2,
            learning_rate: float = 1e-5,
            max_seq_len: Optional[int] = None,
            warmup_proportion: float = 0.2,
            dev_split: float = 0,
            evaluate_every: int = 300,
            save_dir: Optional[str] = None,
            num_processes: Optional[int] = None,
            use_amp: str = None,
    ):
        """
        Fine-tune a model on a TextClassification dataset.
        The dataset needs to be in tabular format (CSV, TSV, etc.), with columns called "label" and "text" in no specific order.
        Options:

        - Take a plain language model (e.g. `bert-base-cased`) and train it for TextClassification
        - Take a TextClassification model and fine-tune it for your domain

        :param data_dir: Path to directory containing your training data
        :param label_list: list of labels in the training dataset, e.g., ["0", "1"]
        :param delimiter: delimiter that separates columns in the training dataset, e.g., "\t"
        :param metric: evaluation metric to be used while training, e.g., "f1_macro"
        :param train_filename: Filename of training data
        :param dev_filename: Filename of dev / eval data
        :param test_filename: Filename of test data
        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                          that gets split off from training data for eval.
        :param use_gpu: Whether to use GPU (if available)
        :param batch_size: Number of samples the model receives in one batch for training
        :param n_epochs: Number of iterations on the whole training data set
        :param learning_rate: Learning rate of the optimizer
        :param max_seq_len: Maximum text length (in tokens). Everything longer gets cut down.
        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.
                                  Until that point LR is increasing linearly. After that it's decreasing again linearly.
                                  Options for different schedules are available in FARM.
        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset
        :param save_dir: Path to store the final model
        :param num_processes: The number of processes for `multiprocessing.Pool` during preprocessing.
                              Set to value of 1 to disable multiprocessing. When set to 1, you cannot split away a dev set from train set.
                              Set to None to use all CPU cores minus one.
        :param use_amp: Optimization level of NVIDIA's automatic mixed precision (AMP). The higher the level, the faster the model.
                        Available options:
                        None (Don't use AMP)
                        "O0" (Normal FP32 training)
                        "O1" (Mixed Precision => Recommended)
                        "O2" (Almost FP16)
                        "O3" (Pure FP16).
                        See details on: https://nvidia.github.io/apex/amp.html
        :return: None
        """

        if dev_filename:
            dev_split = 0

        if num_processes is None:
            num_processes = multiprocessing.cpu_count() - 1 or 1

        set_all_seeds(seed=42)

        # For these variables, by default, we use the value set when initializing the FARMClassifier.
        # These can also be manually set when train() is called if you want a different value at train vs inference
        if use_gpu is None:
            use_gpu = self.use_gpu
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        device, n_gpu = initialize_device_settings(use_cuda=use_gpu, use_amp=use_amp)

        if not save_dir:
            save_dir = f"saved_models/{self.inferencer.model.language_model.name}"

        # 1. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        processor = TextClassificationProcessor(
            tokenizer=self.inferencer.processor.tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metric=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            dev_split=dev_split,
            test_filename=test_filename,
            data_dir=Path(data_dir),
            delimiter=delimiter,
        )

        # 2. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them
        # and calculates a few descriptive statistics of our datasets
        data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False, max_processes=num_processes)

        # 3. Create an optimizer and pass the already initialized model
        model, optimizer, lr_schedule = initialize_optimizer(
            model=self.inferencer.model,
            learning_rate=learning_rate,
            schedule_opts={"name": "LinearWarmup", "warmup_proportion": warmup_proportion},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            device=device,
            use_amp=use_amp,
        )
        # 4. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device,
            use_amp=use_amp,
            disable_tqdm=not self.progress_bar
        )

        # 5. Let it grow!
        self.inferencer.model = trainer.train()
        self.save(Path(save_dir))

    def update_parameters(
            self,
            max_seq_len: Optional[int] = None,
    ):
        """
        Hot update parameters of a loaded FARMClassifier. It may not to be safe when processing concurrent requests.
        """
        if max_seq_len is not None:
            self.inferencer.processor.max_seq_len = max_seq_len
            self.max_seq_len = max_seq_len

    def save(self, directory: Path):
        """
        Saves the FARMClassifier model so that it can be reused at a later point in time.

        :param directory: Directory where the FARMClassifier model should be saved
        """
        logger.info(f"Saving classifier model to {directory}")
        self.inferencer.model.save(directory)
        self.inferencer.processor.save(directory)

    def predict_batch(self, query_doc_list: List[dict], top_k: int = None, batch_size: int = None):
        """
        Use loaded FARMClassifier model to, for a list of queries, classify each query's supplied list of Document.

        Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

        :param query_doc_list: List of dictionaries containing queries with their retrieved documents
        :param top_k: The maximum number of answers to return for each query
        :param batch_size: Number of samples the model receives in one batch for inference
        :return: List of dictionaries containing query and list of Document with class probabilities in meta field
        """
        raise NotImplementedError

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Use loaded classification model to classify the supplied list of Document.

        Returns list of Document enriched with class label and probability, which are stored in Document.meta["classification"]

        :param query: Query string (is not used at the moment)
        :param documents: List of Document to be classified
        :param top_k: The maximum number of documents to return
        :return: List of Document with class probabilities in meta field
        """
        if top_k is None:
            top_k = self.top_k

        # documents should follow the structure {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein ... sei"},
        docs = [{"text": doc.text} for doc in documents]
        results = self.inferencer.inference_from_dicts(dicts=docs)[0]["predictions"]

        classified_docs: List[Document] = []

        for result, doc in zip(results, documents):
            cur_doc = doc
            cur_doc.meta["classification"] = result
            classified_docs.append(cur_doc)

        return classified_docs[:top_k]
