import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextPairClassificationProcessor
from farm.infer import Inferencer
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

from haystack import Document
from haystack.ranker.base import BaseRanker

logger = logging.getLogger(__name__)


class FARMRanker(BaseRanker):
    """
    Transformer based model for Document Re-ranking using the TextPairClassifier of FARM framework (https://github.com/deepset-ai/FARM).
    While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

    |  With a FARMRanker, you can:

     - directly get predictions via predict()
     - fine-tune the model on TextPair data via train()
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
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
        'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
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
        Fine-tune a model on a TextPairClassification dataset. Options:

        - Take a plain language model (e.g. `bert-base-cased`) and train it for TextPairClassification
        - Take a TextPairClassification model and fine-tune it for your domain

        :param data_dir: Path to directory containing your training data in SQuAD style
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

        # For these variables, by default, we use the value set when initializing the FARMRanker.
        # These can also be manually set when train() is called if you want a different value at train vs inference
        if use_gpu is None:
            use_gpu = self.use_gpu
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        device, n_gpu = initialize_device_settings(use_cuda=use_gpu, use_amp=use_amp)

        if not save_dir:
            save_dir = f"saved_models/{self.inferencer.model.language_model.name}"

        # 1. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        label_list = ["0", "1"]
        metric = "f1_macro"
        processor = TextPairClassificationProcessor(
            tokenizer=self.inferencer.processor.tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metric=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            dev_split=dev_split,
            test_filename=test_filename,
            data_dir=Path(data_dir),
            delimiter="\t"
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
        Hot update parameters of a loaded Ranker. It may not to be safe when processing concurrent requests.
        """
        if max_seq_len is not None:
            self.inferencer.processor.max_seq_len = max_seq_len
            self.max_seq_len = max_seq_len

    def save(self, directory: Path):
        """
        Saves the Ranker model so that it can be reused at a later point in time.

        :param directory: Directory where the Ranker model should be saved
        """
        logger.info(f"Saving ranker model to {directory}")
        self.inferencer.model.save(directory)
        self.inferencer.processor.save(directory)

    def predict_batch(self, query_doc_list: List[dict], top_k: int = None, batch_size: int = None):
        """
        Use loaded Ranker model to, for a list of queries, rank each query's supplied list of Document.

        Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

        :param query_doc_list: List of dictionaries containing queries with their retrieved documents
        :param top_k: The maximum number of answers to return for each query
        :param batch_size: Number of samples the model receives in one batch for inference
        :return: List of dictionaries containing query and ranked list of Document
        """
        raise NotImplementedError

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded ranker model to re-rank the supplied list of Document.

        Returns list of Document sorted by (desc.) TextPairClassification similarity with the query.

        :param query: Query string
        :param documents: List of Document to be re-ranked
        :param top_k: The maximum number of documents to return
        :return: List of Document
        """
        if top_k is None:
            top_k = self.top_k

        # calculate similarity of query and each document
        query_and_docs = [{"text": (query, doc.text)} for doc in documents]
        result = self.inferencer.inference_from_dicts(dicts=query_and_docs)
        similarity_scores = [pred["probability"] if pred["label"] == "1" else 1-pred["probability"] for preds in result for pred in preds["predictions"]]

        # rank documents according to scores
        sorted_scores_and_documents = sorted(zip(similarity_scores, documents),
                                             key=lambda similarity_document_tuple: similarity_document_tuple[0],
                                             reverse=True)
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]
