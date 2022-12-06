from typing import List, Optional, Dict, Any, Union, Callable, Tuple

import logging
import multiprocessing
from pathlib import Path
from collections import defaultdict
import os
import tempfile
from time import perf_counter

import torch
from huggingface_hub import create_repo, HfFolder, Repository

from haystack.errors import HaystackError
from haystack.modeling.data_handler.data_silo import DataSilo, DistillationDataSilo
from haystack.modeling.data_handler.processor import SquadProcessor, Processor
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.inputs import QAInput, Question
from haystack.modeling.infer import QAInferencer
from haystack.modeling.model.optimization import initialize_optimizer
from haystack.modeling.model.predictions import QAPred, QACandidate
from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.training import Trainer, DistillationTrainer, TinyBERTDistillationTrainer
from haystack.modeling.evaluation import Evaluator
from haystack.modeling.utils import set_all_seeds, initialize_device_settings

from haystack.schema import Document, Answer, Span
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes.reader.base import BaseReader
from haystack.utils.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)


class FARMReader(BaseReader):
    """
    Transformer based model for extractive Question Answering using the FARM framework (https://github.com/deepset-ai/FARM).
    While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

    With a FARMReader, you can:

     - directly get predictions via predict()
     - fine-tune the model on QA data via train()
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_version: Optional[str] = None,
        context_window_size: int = 150,
        batch_size: int = 50,
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        no_ans_boost: float = 0.0,
        return_no_answer: bool = False,
        top_k: int = 10,
        top_k_per_candidate: int = 3,
        top_k_per_sample: int = 1,
        num_processes: Optional[int] = None,
        max_seq_len: int = 256,
        doc_stride: int = 128,
        progress_bar: bool = True,
        duplicate_filtering: int = 0,
        use_confidence_scores: bool = True,
        confidence_threshold: Optional[float] = None,
        proxies: Optional[Dict[str, str]] = None,
        local_files_only=False,
        force_download=False,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):

        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
        'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param context_window_size: The size, in characters, of the window around the answer span that is used when
                                    displaying the context around the answer.
        :param batch_size: Number of samples the model receives in one batch for inference.
                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                           to a value so only a single batch is used.
        :param use_gpu: Whether to use GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
        If set to 0 (default), the no_answer logit is not changed.
        If a negative number, there is a lower chance of "no_answer" being predicted.
        If a positive number, there is an increased chance of "no_answer"
        :param return_no_answer: Whether to include no_answer predictions in the results.
        :param top_k: The maximum number of answers to return
        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        and that FARM includes no_answer in the sorted list of predictions.
        :param top_k_per_sample: How many answers to extract from each small text passage that the model can process at once
        (one "candidate doc" is usually split into many smaller "passages").
        You usually want a very small value here, as it slows down inference
        and you don't gain much of quality by having multiple answers from one passage.
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        and that FARM includes no_answer in the sorted list of predictions.
        :param num_processes: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer determine optimum number. If you
                              want to debug the Language Model, you might need to disable multiprocessing!
        :param max_seq_len: Max sequence length of one input text for the model
        :param doc_stride: Length of striding window for splitting long texts (used if ``len(text) > max_seq_len``)
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.
                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
        :param use_confidence_scores: Determines the type of score that is used for ranking a predicted answer.
                                      `True` => a scaled confidence / relevance score between [0, 1].
                                      This score can also be further calibrated on your dataset via self.eval()
                                      (see https://docs.haystack.deepset.ai/docs/reader#confidence-scores).
                                      `False` => an unscaled, raw score [-inf, +inf] which is the sum of start and end logit
                                      from the model for the predicted span.
                                      Using confidence scores can change the ranking of no_answer compared to using the
                                      unscaled raw scores.
        :param confidence_threshold: Filters out predictions below confidence_threshold. Value should be between 0 and 1. Disabled by default.
        :param proxies: Dict of proxy servers to use for downloading external models. Example: {'http': 'some.proxy:1234', 'http://hostname': 'my.proxy:3111'}
        :param local_files_only: Whether to force checking for local files only (and forbid downloads)
        :param force_download: Whether fo force a (re-)download even if the model exists locally in the cache.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        self.devices, self.n_gpu = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)
        self.return_no_answers = return_no_answer
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.inferencer = QAInferencer.load(
            model_name_or_path,
            batch_size=batch_size,
            gpu=self.n_gpu > 0,
            task_type="question_answering",
            max_seq_len=max_seq_len,
            doc_stride=doc_stride,
            num_processes=num_processes,
            revision=model_version,
            disable_tqdm=not progress_bar,
            strict=False,
            proxies=proxies,
            local_files_only=local_files_only,
            force_download=force_download,
            devices=self.devices,
            use_auth_token=use_auth_token,
        )
        self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        self.inferencer.model.prediction_heads[0].n_best = top_k_per_candidate + 1  # including possible no_answer
        self.inferencer.model.prediction_heads[0].n_best_per_sample = top_k_per_sample
        self.inferencer.model.prediction_heads[0].duplicate_filtering = duplicate_filtering
        self.inferencer.model.prediction_heads[0].use_confidence_scores_for_ranking = use_confidence_scores
        self.max_seq_len = max_seq_len
        self.progress_bar = progress_bar
        self.use_confidence_scores = use_confidence_scores
        self.confidence_threshold = confidence_threshold
        self.model_name_or_path = model_name_or_path  # Used in distillation, see DistillationDataSilo._get_checksum()

    def _training_procedure(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        devices: List[torch.device] = [],
        batch_size: int = 10,
        n_epochs: int = 2,
        learning_rate: float = 1e-5,
        max_seq_len: Optional[int] = None,
        warmup_proportion: float = 0.2,
        dev_split: float = 0,
        evaluate_every: int = 300,
        save_dir: Optional[str] = None,
        num_processes: Optional[int] = None,
        use_amp: Optional[str] = None,
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
        teacher_model: Optional["FARMReader"] = None,
        teacher_batch_size: Optional[int] = None,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
        distillation_loss_weight: float = 0.5,
        distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "kl_div",
        temperature: float = 1.0,
        tinybert: bool = False,
        processor: Optional[Processor] = None,
        grad_acc_steps: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        if dev_filename:
            dev_split = 0

        if num_processes is None:
            num_processes = multiprocessing.cpu_count() - 1 or 1

        set_all_seeds(seed=42)

        # For these variables, by default, we use the value set when initializing the FARMReader.
        # These can also be manually set when train() is called if you want a different value at train vs inference
        if devices is None:
            devices = self.devices
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        devices, n_gpu = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)

        if not save_dir:
            save_dir = f"../../saved_models/{self.inferencer.model.language_model.name}"
            if tinybert:
                save_dir += "_tinybert_stage_1"

        # 1. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        label_list = ["start_token", "end_token"]
        metric = "squad"
        if processor is None:
            processor = SquadProcessor(
                tokenizer=self.inferencer.processor.tokenizer,
                max_seq_len=max_seq_len,
                label_list=label_list,
                metric=metric,
                train_filename=train_filename,
                dev_filename=dev_filename,
                dev_split=dev_split,
                test_filename=test_filename,
                data_dir=Path(data_dir),
            )
        data_silo: DataSilo

        # 2. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them
        # and calculates a few descriptive statistics of our datasets
        if (
            teacher_model and not tinybert
        ):  # checks if teacher model is passed as parameter, in that case assume model distillation is used
            data_silo = DistillationDataSilo(
                teacher_model,
                teacher_batch_size or batch_size,
                device=devices[0],
                processor=processor,
                batch_size=batch_size,
                distributed=False,
                max_processes=num_processes,
                caching=caching,
                cache_path=cache_path,
            )
        else:  # caching would need too much memory for tinybert distillation so in that case we use the default data silo
            data_silo = DataSilo(
                processor=processor,
                batch_size=batch_size,
                distributed=False,
                max_processes=num_processes,
                caching=caching,
                cache_path=cache_path,
            )

        # 3. Create an optimizer and pass the already initialized model
        model, optimizer, lr_schedule = initialize_optimizer(
            model=self.inferencer.model,
            # model=self.inferencer.model,
            learning_rate=learning_rate,
            schedule_opts={"name": "LinearWarmup", "warmup_proportion": warmup_proportion},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            device=devices[0],
            use_amp=use_amp,
            grad_acc_steps=grad_acc_steps,
        )
        # 4. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        if tinybert:
            if not teacher_model:
                raise ValueError("TinyBERT distillation requires a teacher model.")
            trainer = TinyBERTDistillationTrainer.create_or_load_checkpoint(
                model=model,
                teacher_model=teacher_model.inferencer.model,  # teacher needs to be passed as teacher outputs aren't cached
                optimizer=optimizer,
                data_silo=data_silo,
                epochs=n_epochs,
                n_gpu=n_gpu,
                lr_schedule=lr_schedule,
                evaluate_every=evaluate_every,
                device=devices[0],
                use_amp=use_amp,
                disable_tqdm=not self.progress_bar,
                checkpoint_root_dir=Path(checkpoint_root_dir),
                checkpoint_every=checkpoint_every,
                checkpoints_to_keep=checkpoints_to_keep,
                grad_acc_steps=grad_acc_steps,
                early_stopping=early_stopping,
            )

        elif (
            teacher_model
        ):  # checks again if teacher model is passed as parameter, in that case assume model distillation is used
            trainer = DistillationTrainer.create_or_load_checkpoint(
                model=model,
                optimizer=optimizer,
                data_silo=data_silo,
                epochs=n_epochs,
                n_gpu=n_gpu,
                lr_schedule=lr_schedule,
                evaluate_every=evaluate_every,
                device=devices[0],
                use_amp=use_amp,
                disable_tqdm=not self.progress_bar,
                checkpoint_root_dir=Path(checkpoint_root_dir),
                checkpoint_every=checkpoint_every,
                checkpoints_to_keep=checkpoints_to_keep,
                distillation_loss=distillation_loss,
                distillation_loss_weight=distillation_loss_weight,
                temperature=temperature,
                grad_acc_steps=grad_acc_steps,
                early_stopping=early_stopping,
            )
        else:
            trainer = Trainer.create_or_load_checkpoint(
                model=model,
                optimizer=optimizer,
                data_silo=data_silo,
                epochs=n_epochs,
                n_gpu=n_gpu,
                lr_schedule=lr_schedule,
                evaluate_every=evaluate_every,
                device=devices[0],
                use_amp=use_amp,
                disable_tqdm=not self.progress_bar,
                checkpoint_root_dir=Path(checkpoint_root_dir),
                checkpoint_every=checkpoint_every,
                checkpoints_to_keep=checkpoints_to_keep,
                grad_acc_steps=grad_acc_steps,
                early_stopping=early_stopping,
            )

        # 5. Let it grow!
        self.inferencer.model = trainer.train()
        self.save(Path(save_dir))

    def train(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        devices: List[torch.device] = [],
        batch_size: int = 10,
        n_epochs: int = 2,
        learning_rate: float = 1e-5,
        max_seq_len: Optional[int] = None,
        warmup_proportion: float = 0.2,
        dev_split: float = 0,
        evaluate_every: int = 300,
        save_dir: Optional[str] = None,
        num_processes: Optional[int] = None,
        use_amp: Optional[str] = None,
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
        grad_acc_steps: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """
        Fine-tune a model on a QA dataset. Options:
        - Take a plain language model (e.g. `bert-base-cased`) and train it for QA (e.g. on SQuAD data)
        - Take a QA model (e.g. `deepset/bert-base-cased-squad2`) and fine-tune it for your domain (e.g. using your labels collected via the haystack annotation tool)

        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

        Note that when performing training with this function, long documents are split into chunks.
        If a chunk doesn't contain the answer to the question, it is treated as a no-answer sample.

        :param data_dir: Path to directory containing your training data in SQuAD style
        :param train_filename: Filename of training data
        :param dev_filename: Filename of dev / eval data
        :param test_filename: Filename of test data
        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                          that gets split off from training data for eval.
        :param use_gpu: Whether to use GPU (if available)
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param batch_size: Number of samples the model receives in one batch for training
        :param n_epochs: Number of iterations on the whole training data set
        :param learning_rate: Learning rate of the optimizer
        :param max_seq_len: Maximum text length (in tokens). Everything longer gets cut down.
        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.
                                  Until that point LR is increasing linearly. After that it's decreasing again linearly.
                                  Options for different schedules are available in FARM.
        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset.
                               Note that the evaluation report is logged at evaluation level INFO while Haystack's default is WARNING.
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
        :param checkpoint_root_dir: The Path of a directory where all train checkpoints are saved. For each individual
               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
        :param checkpoint_every: Save a train checkpoint after this many steps of training.
        :param checkpoints_to_keep: The maximum number of train checkpoints to save.
        :param caching: Whether or not to use caching for the preprocessed dataset.
        :param cache_path: The Path to cache the preprocessed dataset.
        :param grad_acc_steps: The number of steps to accumulate gradients for before performing a backward pass.
        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.
        :return: None
        """
        return self._training_procedure(
            data_dir=data_dir,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            use_gpu=use_gpu,
            devices=devices,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            max_seq_len=max_seq_len,
            warmup_proportion=warmup_proportion,
            dev_split=dev_split,
            evaluate_every=evaluate_every,
            save_dir=save_dir,
            num_processes=num_processes,
            use_amp=use_amp,
            checkpoint_root_dir=checkpoint_root_dir,
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
            caching=caching,
            cache_path=cache_path,
            grad_acc_steps=grad_acc_steps,
            early_stopping=early_stopping,
        )

    def distil_prediction_layer_from(
        self,
        teacher_model: "FARMReader",
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        devices: List[torch.device] = [],
        student_batch_size: int = 10,
        teacher_batch_size: Optional[int] = None,
        n_epochs: int = 2,
        learning_rate: float = 3e-5,
        max_seq_len: Optional[int] = None,
        warmup_proportion: float = 0.2,
        dev_split: float = 0,
        evaluate_every: int = 300,
        save_dir: Optional[str] = None,
        num_processes: Optional[int] = None,
        use_amp: Optional[str] = None,
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
        distillation_loss_weight: float = 0.5,
        distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "kl_div",
        temperature: float = 1.0,
        grad_acc_steps: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """
        Fine-tune a model on a QA dataset using logit-based distillation. You need to provide a teacher model that is already finetuned on the dataset
        and a student model that will be trained using the teacher's logits. The idea of this is to increase the accuracy of a lightweight student model.
        using a more complex teacher.
        Originally proposed in: https://arxiv.org/pdf/1503.02531.pdf
        This can also be considered as the second stage of distillation finetuning as described in the TinyBERT paper:
        https://arxiv.org/pdf/1909.10351.pdf
        **Example**
        ```python
        student = FARMReader(model_name_or_path="prajjwal1/bert-medium")
        teacher = FARMReader(model_name_or_path="deepset/bert-large-uncased-whole-word-masking-squad2")
        student.distil_prediction_layer_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",
                            learning_rate=3e-5, distillation_loss_weight=1.0, temperature=5)
        ```

        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

        :param teacher_model: Model whose logits will be used to improve accuracy
        :param data_dir: Path to directory containing your training data in SQuAD style
        :param train_filename: Filename of training data
        :param dev_filename: Filename of dev / eval data
        :param test_filename: Filename of test data
        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                          that gets split off from training data for eval.
        :param use_gpu: Whether to use GPU (if available)
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param student_batch_size: Number of samples the student model receives in one batch for training
        :param student_batch_size: Number of samples the teacher model receives in one batch for distillation
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
        :param checkpoint_root_dir: the Path of directory where all train checkpoints are saved. For each individual
               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
        :param checkpoint_every: save a train checkpoint after this many steps of training.
        :param checkpoints_to_keep: maximum number of train checkpoints to save.
        :param caching: whether or not to use caching for preprocessed dataset and teacher logits
        :param cache_path: Path to cache the preprocessed dataset and teacher logits
        :param distillation_loss_weight: The weight of the distillation loss. A higher weight means the teacher outputs are more important.
        :param distillation_loss: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named parameters student_logits and teacher_logits)
        :param temperature: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.
        :param tinybert_loss: Whether to use the TinyBERT loss function for distillation. This requires the student to be a TinyBERT model and the teacher to be a finetuned version of bert-base-uncased.
        :param tinybert_epochs: Number of epochs to train the student model with the TinyBERT loss function. After this many epochs, the student model is trained with the regular distillation loss function.
        :param tinybert_learning_rate: Learning rate to use when training the student model with the TinyBERT loss function.
        :param tinybert_train_filename: Filename of training data to use when training the student model with the TinyBERT loss function. To best follow the original paper, this should be an augmented version of the training data created using the augment_squad.py script. If not specified, the training data from the original training is used.
        :param processor: The processor to use for preprocessing. If None, the default SquadProcessor is used.
        :param grad_acc_steps: The number of steps to accumulate gradients for before performing a backward pass.
        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.
        :return: None
        """
        return self._training_procedure(
            data_dir=data_dir,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            use_gpu=use_gpu,
            devices=devices,
            batch_size=student_batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            max_seq_len=max_seq_len,
            warmup_proportion=warmup_proportion,
            dev_split=dev_split,
            evaluate_every=evaluate_every,
            save_dir=save_dir,
            num_processes=num_processes,
            use_amp=use_amp,
            checkpoint_root_dir=checkpoint_root_dir,
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
            teacher_model=teacher_model,
            teacher_batch_size=teacher_batch_size,
            caching=caching,
            cache_path=cache_path,
            distillation_loss_weight=distillation_loss_weight,
            distillation_loss=distillation_loss,
            temperature=temperature,
            grad_acc_steps=grad_acc_steps,
            early_stopping=early_stopping,
        )

    def distil_intermediate_layers_from(
        self,
        teacher_model: "FARMReader",
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        devices: List[torch.device] = [],
        batch_size: int = 10,
        n_epochs: int = 5,
        learning_rate: float = 5e-5,
        max_seq_len: Optional[int] = None,
        warmup_proportion: float = 0.2,
        dev_split: float = 0,
        evaluate_every: int = 300,
        save_dir: Optional[str] = None,
        num_processes: Optional[int] = None,
        use_amp: Optional[str] = None,
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
        distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "mse",
        temperature: float = 1.0,
        processor: Optional[Processor] = None,
        grad_acc_steps: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """
        The first stage of distillation finetuning as described in the TinyBERT paper:
        https://arxiv.org/pdf/1909.10351.pdf
        **Example**
        ```python
        student = FARMReader(model_name_or_path="prajjwal1/bert-medium")
        teacher = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_6L_768D")
        student.distil_intermediate_layers_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",
                            learning_rate=3e-5, distillation_loss_weight=1.0, temperature=5)
        ```

        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.

        :param teacher_model: Model whose logits will be used to improve accuracy
        :param data_dir: Path to directory containing your training data in SQuAD style
        :param train_filename: Filename of training data. To best follow the original paper, this should be an augmented version of the training data created using the augment_squad.py script
        :param dev_filename: Filename of dev / eval data
        :param test_filename: Filename of test data
        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                          that gets split off from training data for eval.
        :param use_gpu: Whether to use GPU (if available)
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param student_batch_size: Number of samples the student model receives in one batch for training
        :param student_batch_size: Number of samples the teacher model receives in one batch for distillation
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
        :param checkpoint_root_dir: the Path of directory where all train checkpoints are saved. For each individual
               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
        :param checkpoint_every: save a train checkpoint after this many steps of training.
        :param checkpoints_to_keep: maximum number of train checkpoints to save.
        :param caching: whether or not to use caching for preprocessed dataset and teacher logits
        :param cache_path: Path to cache the preprocessed dataset and teacher logits
        :param distillation_loss_weight: The weight of the distillation loss. A higher weight means the teacher outputs are more important.
        :param distillation_loss: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named parameters student_logits and teacher_logits)
        :param temperature: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.
        :param processor: The processor to use for preprocessing. If None, the default SquadProcessor is used.
        :param grad_acc_steps: The number of steps to accumulate gradients for before performing a backward pass.
        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.
        :return: None
        """
        return self._training_procedure(
            data_dir=data_dir,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            use_gpu=use_gpu,
            devices=devices,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            max_seq_len=max_seq_len,
            warmup_proportion=warmup_proportion,
            dev_split=dev_split,
            evaluate_every=evaluate_every,
            save_dir=save_dir,
            num_processes=num_processes,
            use_amp=use_amp,
            checkpoint_root_dir=checkpoint_root_dir,
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
            teacher_model=teacher_model,
            teacher_batch_size=batch_size,
            caching=caching,
            cache_path=cache_path,
            distillation_loss=distillation_loss,
            temperature=temperature,
            tinybert=True,
            processor=processor,
            grad_acc_steps=grad_acc_steps,
            early_stopping=early_stopping,
        )

    def update_parameters(
        self,
        context_window_size: Optional[int] = None,
        no_ans_boost: Optional[float] = None,
        return_no_answer: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
        doc_stride: Optional[int] = None,
    ):
        """
        Hot update parameters of a loaded Reader. It may not to be safe when processing concurrent requests.
        """
        if no_ans_boost is not None:
            self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        if return_no_answer is not None:
            self.return_no_answers = return_no_answer
        if doc_stride is not None:
            self.inferencer.processor.doc_stride = doc_stride
        if context_window_size is not None:
            self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        if max_seq_len is not None:
            self.inferencer.processor.max_seq_len = max_seq_len
            self.max_seq_len = max_seq_len

    def save(self, directory: Path):
        """
        Saves the Reader model so that it can be reused at a later point in time.

        :param directory: Directory where the Reader model should be saved
        """
        logger.info("Saving reader model to %s", directory)
        self.inferencer.model.save(directory)
        self.inferencer.processor.save(directory)

    def save_to_remote(
        self, repo_id: str, private: Optional[bool] = None, commit_message: str = "Add new model to Hugging Face."
    ):
        """
        Saves the Reader model to Hugging Face Model Hub with the given model_name. For this to work:
        - Be logged in to Hugging Face on your machine via transformers-cli
        - Have git lfs installed (https://packagecloud.io/github/git-lfs/install), you can test it by git lfs --version

        :param repo_id: A namespace (user or an organization) and a repo name separated by a '/' of the model you want to save to Hugging Face
        :param private: Set to true to make the model repository private
        :param commit_message: Commit message while saving to Hugging Face
        """
        # Note: This function was inspired by the save_to_hub function in the sentence-transformers repo (https://github.com/UKPLab/sentence-transformers/)
        # Especially for git-lfs tracking.

        token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "To save this reader model to Hugging Face, make sure you login to the hub on this computer by typing `transformers-cli login`."
            )

        repo_url = create_repo(token=token, repo_id=repo_id, private=private, repo_type=None, exist_ok=True)

        transformer_models = self.inferencer.model.convert_to_transformers()

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Repository(tmp_dir, clone_from=repo_url)

            self.inferencer.processor.tokenizer.save_pretrained(tmp_dir)

            # convert_to_transformers (above) creates one model per prediction head.
            # As the FarmReader models only have one head (QA) we go with this.
            transformer_models[0].save_pretrained(tmp_dir)

            large_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, tmp_dir)

                    if os.path.getsize(file_path) > (5 * 1024 * 1024):
                        large_files.append(rel_path)

            if len(large_files) > 0:
                logger.info("Track files with git lfs: {}".format(", ".join(large_files)))
                repo.lfs_track(large_files)

            logger.info("Push model to the hub. This might take a while")
            commit_url = repo.push_to_hub(commit_message=commit_message)

        return commit_url

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Use loaded QA model to find answers for the queries in the Documents.

        - If you provide a list containing a single query...

            - ... and a single list of Documents, the query will be applied to each Document individually.
            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
              will be aggregated per Document list.

        - If you provide a list of multiple queries...

            - ... and a single list of Documents, each query will be applied to each Document individually.
            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
              and the Answers will be aggregated per query-Document pair.

        :param queries: Single query or list of queries.
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
                          Can be a single list of Documents or a list of lists of Documents.
        :param top_k: Number of returned answers per query.
        :param batch_size: Number of query-document pairs to be processed at a time.
        """
        if top_k is None:
            top_k = self.top_k

        inputs, number_of_docs, single_doc_list = self._preprocess_batch_queries_and_docs(
            queries=queries, documents=documents
        )

        if batch_size is not None:
            self.inferencer.batch_size = batch_size
        # Make predictions on all document-query pairs
        predictions = self.inferencer.inference_from_objects(
            objects=inputs, return_json=False, multiprocessing_chunksize=10
        )

        # Group predictions together
        grouped_predictions = []
        left_idx = 0
        right_idx = 0
        for number in number_of_docs:
            right_idx = left_idx + number
            grouped_predictions.append(predictions[left_idx:right_idx])
            left_idx = right_idx

        results: Dict = {"queries": queries, "answers": [], "no_ans_gaps": []}
        for group in grouped_predictions:
            answers, max_no_ans_gap = self._extract_answers_of_predictions(group, top_k)
            results["answers"].append(answers)
            results["no_ans_gaps"].append(max_no_ans_gap)

        # Group answers by question in case of multiple queries and single doc list
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results["answers"]) / len(queries))
            answers = []
            for i in range(0, len(results["answers"]), answers_per_query):
                answer_group = results["answers"][i : i + answers_per_query]
                answers.append(answer_group)
            results["answers"] = answers

        return results

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a query in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) score.
        Example:

        ```python
        {
            'query': 'Who is the father of Arya Stark?',
            'answers':[Answer(
                         'answer': 'Eddard,',
                         'context': "She travels with her father, Eddard, to King's Landing when he is",
                         'score': 0.9787139466668613,
                         'offsets_in_context': [Span(start=29, end=35],
                         'offsets_in_context': [Span(start=347, end=353],
                         'document_id': '88d1ed769d003939d3a0d28034464ab2'
                         ),...
                      ]
        }
         ```

        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k
        # convert input to FARM format
        inputs = []
        for doc in documents:
            cur = QAInput(doc_text=doc.content, questions=Question(text=query, uid=doc.id))
            inputs.append(cur)

        # get answers from QA model
        # TODO: Need fix in FARM's `to_dict` function of `QAInput` class
        predictions = self.inferencer.inference_from_objects(
            objects=inputs, return_json=False, multiprocessing_chunksize=1
        )
        # assemble answers from all the different documents & format them.
        answers, max_no_ans_gap = self._extract_answers_of_predictions(predictions, top_k)
        # TODO: potentially simplify return here to List[Answer] and handle no_ans_gap differently
        result = {"query": query, "no_ans_gap": max_no_ans_gap, "answers": answers}

        return result

    def eval_on_file(
        self,
        data_dir: Union[Path, str],
        test_filename: str,
        device: Optional[Union[str, torch.device]] = None,
        calibrate_conf_scores: bool = False,
    ):
        """
        Performs evaluation on a SQuAD-formatted file.
        Returns a dict containing the following metrics:
            - "EM": exact match score
            - "f1": F1-Score
            - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

        :param data_dir: The directory in which the test set can be found
        :param test_filename: The name of the file containing the test data in SQuAD format.
        :param device: The device on which the tensors should be processed.
               Choose from torch.device("cpu") and torch.device("cuda") (or simply "cpu" or "cuda")
               or use the Reader's device by default.
        :param calibrate_conf_scores: Whether to calibrate the temperature for scaling of the confidence scores.
        """
        logger.warning(
            "FARMReader.eval_on_file() uses a slightly different evaluation approach than `Pipeline.eval()`:\n"
            "- instead of giving you full control over which labels to use, this method always returns three types of metrics: combined (no suffix), text_answer ('_text_answer' suffix) and no_answer ('_no_answer' suffix) metrics.\n"
            "- instead of comparing predictions with labels on a string level, this method compares them on a token-ID level. This makes it unable to do any string normalization (e.g. normalize whitespaces) beforehand.\n"
            "Hence, results might slightly differ from those of `Pipeline.eval()`\n."
            "If you are just about starting to evaluate your model consider using `Pipeline.eval()` instead."
        )

        if device is None:
            device = self.devices[0]
        else:
            device = torch.device(device)

        eval_processor = SquadProcessor(
            tokenizer=self.inferencer.processor.tokenizer,
            max_seq_len=self.inferencer.processor.max_seq_len,
            label_list=self.inferencer.processor.tasks["question_answering"]["label_list"],
            metric=self.inferencer.processor.tasks["question_answering"]["metric"],
            train_filename=None,
            dev_filename=None,
            dev_split=0,
            test_filename=test_filename,
            data_dir=Path(data_dir),
        )

        data_silo = DataSilo(processor=eval_processor, batch_size=self.inferencer.batch_size, distributed=False)
        data_loader = data_silo.get_data_loader("test")

        evaluator = Evaluator(data_loader=data_loader, tasks=eval_processor.tasks, device=device)

        eval_results = evaluator.eval(
            self.inferencer.model,
            calibrate_conf_scores=calibrate_conf_scores,
            use_confidence_scores_for_ranking=self.use_confidence_scores,
        )
        results = {
            "EM": eval_results[0]["EM"] * 100,
            "f1": eval_results[0]["f1"] * 100,
            "top_n_accuracy": eval_results[0]["top_n_accuracy"] * 100,
            "top_n": self.inferencer.model.prediction_heads[0].n_best,
            "EM_text_answer": eval_results[0]["EM_text_answer"] * 100,
            "f1_text_answer": eval_results[0]["f1_text_answer"] * 100,
            "top_n_accuracy_text_answer": eval_results[0]["top_n_accuracy_text_answer"] * 100,
            "top_n_EM_text_answer": eval_results[0]["top_n_EM_text_answer"] * 100,
            "top_n_f1_text_answer": eval_results[0]["top_n_f1_text_answer"] * 100,
            "Total_text_answer": eval_results[0]["Total_text_answer"],
            "EM_no_answer": eval_results[0]["EM_no_answer"] * 100,
            "f1_no_answer": eval_results[0]["f1_no_answer"] * 100,
            "top_n_accuracy_no_answer": eval_results[0]["top_n_accuracy_no_answer"] * 100,
            "Total_no_answer": eval_results[0]["Total_no_answer"],
        }
        return results

    def eval(
        self,
        document_store: BaseDocumentStore,
        device: Optional[Union[str, torch.device]] = None,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold-label",
        calibrate_conf_scores: bool = False,
    ):
        """
        Performs evaluation on evaluation documents in the DocumentStore.
        Returns a dict containing the following metrics:
              - "EM": Proportion of exact matches of predicted answers with their corresponding correct answers
              - "f1": Average overlap between predicted answers and their corresponding correct answers
              - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

        :param document_store: DocumentStore containing the evaluation documents
        :param device: The device on which the tensors should be processed.
                       Choose from torch.device("cpu") and torch.device("cuda") (or simply "cpu" or "cuda")
                       or use the Reader's device by default.
        :param label_index: Index/Table name where labeled questions are stored
        :param doc_index: Index/Table name where documents that are used for evaluation are stored
        :param label_origin: Field name where the gold labels are stored
        :param calibrate_conf_scores: Whether to calibrate the temperature for scaling of the confidence scores.
        """
        logger.warning(
            "FARMReader.eval() uses a slightly different evaluation approach than `Pipeline.eval()`:\n"
            "- instead of giving you full control over which labels to use, this method always returns three types of metrics: combined (no suffix), text_answer ('_text_answer' suffix) and no_answer ('_no_answer' suffix) metrics.\n"
            "- instead of comparing predictions with labels on a string level, this method compares them on a token-ID level. This makes it unable to do any string normalization (e.g. normalize whitespaces) beforehand.\n"
            "Hence, results might slightly differ from those of `Pipeline.eval()`\n."
            "If you are just about starting to evaluate your model consider using `Pipeline.eval()` instead."
        )

        if device is None:
            device = self.devices[0]
        else:
            device = torch.device(device)

        if self.top_k_per_candidate != 4:
            logger.info(
                f"Performing Evaluation using top_k_per_candidate = {self.top_k_per_candidate} \n"
                f"and consequently, QuestionAnsweringPredictionHead.n_best = {self.top_k_per_candidate + 1}. \n"
                f"This deviates from FARM's default where QuestionAnsweringPredictionHead.n_best = 5"
            )

        # extract all questions for evaluation
        filters: Dict = {"origin": [label_origin]}

        labels = document_store.get_all_labels(index=label_index, filters=filters)

        # Aggregate all answer labels per question
        aggregated_per_doc = defaultdict(list)
        for label in labels:
            if not label.document.id:
                logger.error("Label does not contain a document id")
                continue
            aggregated_per_doc[label.document.id].append(label)

        # Create squad style dicts
        d: Dict[str, Any] = {}
        all_doc_ids = [x.id for x in document_store.get_all_documents(doc_index)]
        for doc_id in all_doc_ids:
            doc = document_store.get_document_by_id(doc_id, index=doc_index)
            if not doc:
                logger.error("Document with the ID '%s' is not present in the document store.", doc_id)
                continue
            d[str(doc_id)] = {"context": doc.content}
            # get all questions / answers
            # TODO check if we can simplify this by using MultiLabel
            aggregated_per_question: Dict[tuple, Any] = defaultdict(list)

            if doc_id in aggregated_per_doc:
                for label in aggregated_per_doc[doc_id]:
                    aggregation_key = (doc_id, label.query)
                    if label.answer is None:
                        logger.error("Label.answer was None, but Answer object was expected: %s", label)
                        continue
                    if label.answer.offsets_in_document is None:
                        logger.error(
                            f"Label.answer.offsets_in_document was None, but Span object was expected: {label} "
                        )
                        continue
                    # add to existing answers
                    # TODO offsets (whole block)
                    if aggregation_key in aggregated_per_question.keys():
                        if label.no_answer:
                            continue

                        # Hack to fix problem where duplicate questions are merged by doc_store processing creating a QA example with 8 annotations > 6 annotation max
                        if len(aggregated_per_question[aggregation_key]["answers"]) >= 6:
                            logger.warning(
                                f"Answers in this sample are being dropped because it has more than 6 answers. (doc_id: {doc_id}, question: {label.query}, label_id: {label.id})"
                            )
                            continue
                        aggregated_per_question[aggregation_key]["answers"].append(
                            {"text": label.answer.answer, "answer_start": label.answer.offsets_in_document[0].start}
                        )
                        aggregated_per_question[aggregation_key]["is_impossible"] = False
                    # create new one
                    else:
                        # We don't need to create an answer dict if is_impossible / no_answer
                        if label.no_answer == True:
                            aggregated_per_question[aggregation_key] = {
                                "id": str(hash(str(doc_id) + label.query)),
                                "question": label.query,
                                "answers": [],
                                "is_impossible": True,
                            }
                        else:
                            aggregated_per_question[aggregation_key] = {
                                "id": str(hash(str(doc_id) + label.query)),
                                "question": label.query,
                                "answers": [
                                    {
                                        "text": label.answer.answer,
                                        "answer_start": label.answer.offsets_in_document[0].start,
                                    }
                                ],
                                "is_impossible": False,
                            }

            # Get rid of the question key again (after we aggregated we don't need it anymore)
            d[str(doc_id)]["qas"] = [v for v in aggregated_per_question.values()]

        # Convert input format for FARM
        farm_input = [v for v in d.values()]
        n_queries = len([y for x in farm_input for y in x["qas"]])

        # Create DataLoader that can be passed to the Evaluator
        tic = perf_counter()
        indices = range(len(farm_input))
        dataset, tensor_names, problematic_ids = self.inferencer.processor.dataset_from_dicts(
            farm_input, indices=indices
        )
        data_loader = NamedDataLoader(dataset=dataset, batch_size=self.inferencer.batch_size, tensor_names=tensor_names)

        evaluator = Evaluator(data_loader=data_loader, tasks=self.inferencer.processor.tasks, device=device)

        eval_results = evaluator.eval(
            self.inferencer.model,
            calibrate_conf_scores=calibrate_conf_scores,
            use_confidence_scores_for_ranking=self.use_confidence_scores,
        )
        toc = perf_counter()
        reader_time = toc - tic
        results = {
            "EM": eval_results[0]["EM"] * 100,
            "f1": eval_results[0]["f1"] * 100,
            "top_n_accuracy": eval_results[0]["top_n_accuracy"] * 100,
            "top_n": self.inferencer.model.prediction_heads[0].n_best,
            "reader_time": reader_time,
            "seconds_per_query": reader_time / n_queries,
            "EM_text_answer": eval_results[0]["EM_text_answer"] * 100,
            "f1_text_answer": eval_results[0]["f1_text_answer"] * 100,
            "top_n_accuracy_text_answer": eval_results[0]["top_n_accuracy_text_answer"] * 100,
            "top_n_EM_text_answer": eval_results[0]["top_n_EM_text_answer"] * 100,
            "top_n_f1_text_answer": eval_results[0]["top_n_f1_text_answer"] * 100,
            "Total_text_answer": eval_results[0]["Total_text_answer"],
            "EM_no_answer": eval_results[0]["EM_no_answer"] * 100,
            "f1_no_answer": eval_results[0]["f1_no_answer"] * 100,
            "top_n_accuracy_no_answer": eval_results[0]["top_n_accuracy_no_answer"] * 100,
            "Total_no_answer": eval_results[0]["Total_no_answer"],
        }
        return results

    def _extract_answers_of_predictions(self, predictions: List[QAPred], top_k: Optional[int] = None):
        # Assemble answers from all the different documents and format them.
        # For the 'no answer' option, we collect all no_ans_gaps and decide how likely
        # a no answer is based on all no_ans_gaps values across all documents
        answers: List[Answer] = []
        no_ans_gaps = []
        best_score_answer = 0

        for pred in predictions:
            answers_per_document = []
            no_ans_gaps.append(pred.no_answer_gap)
            for ans in pred.prediction:
                # skip 'no answers' here
                if self._check_no_answer(ans):
                    pass
                else:
                    cur = Answer(
                        answer=ans.answer,
                        type="extractive",
                        score=ans.confidence if self.use_confidence_scores else ans.score,
                        context=ans.context_window,
                        document_id=pred.id,
                        offsets_in_context=[
                            Span(
                                start=ans.offset_answer_start - ans.offset_context_window_start,
                                end=ans.offset_answer_end - ans.offset_context_window_start,
                            )
                        ],
                        offsets_in_document=[Span(start=ans.offset_answer_start, end=ans.offset_answer_end)],
                    )

                    answers_per_document.append(cur)

                    if ans.score > best_score_answer:
                        best_score_answer = ans.score

            # Only take n best candidates. Answers coming back from FARM are sorted with decreasing relevance
            answers += answers_per_document[: self.top_k_per_candidate]

        # calculate the score for predicting 'no answer', relative to our best positive answer score
        no_ans_prediction, max_no_ans_gap = self._calc_no_answer(
            no_ans_gaps, best_score_answer, self.use_confidence_scores
        )
        if self.return_no_answers:
            answers.append(no_ans_prediction)

        # sort answers by score (descending) and select top-k
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        # apply confidence based filtering if enabled
        if self.confidence_threshold is not None:
            answers = [ans for ans in answers if ans.score is not None and ans.score >= self.confidence_threshold]

        return answers, max_no_ans_gap

    def _preprocess_batch_queries_and_docs(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]]
    ) -> Tuple[List[QAInput], List[int], bool]:
        # Convert input to FARM format
        inputs = []
        number_of_docs = []
        single_doc_list = False

        # Docs case 1: single list of Documents -> apply each query to all Documents
        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
            for query in queries:
                for doc in documents:
                    number_of_docs.append(1)
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    cur = QAInput(doc_text=doc.content, questions=Question(text=query, uid=doc.id))
                    inputs.append(cur)

        # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents, if queries
        # contains only one query, apply it to each list of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            single_doc_list = False
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                number_of_docs.append(len(cur_docs))
                for doc in cur_docs:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    cur = QAInput(doc_text=doc.content, questions=Question(text=query, uid=doc.id))
                    inputs.append(cur)

        return inputs, number_of_docs, single_doc_list

    def calibrate_confidence_scores(
        self,
        document_store: BaseDocumentStore,
        device: Optional[Union[str, torch.device]] = None,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
    ):
        """
        Calibrates confidence scores on evaluation documents in the DocumentStore.

        :param document_store: DocumentStore containing the evaluation documents
        :param device: The device on which the tensors should be processed.
                       Choose from torch.device("cpu") and torch.device("cuda") (or simply "cpu" or "cuda")
                       or use the Reader's device by default.
        :param label_index: Index/Table name where labeled questions are stored
        :param doc_index: Index/Table name where documents that are used for evaluation are stored
        :param label_origin: Field name where the gold labels are stored
        """
        if device is None:
            device = self.devices[0]
        self.eval(
            document_store=document_store,
            device=device,
            label_index=label_index,
            doc_index=doc_index,
            label_origin=label_origin,
            calibrate_conf_scores=True,
        )

    @staticmethod
    def _check_no_answer(c: QACandidate):
        # check for correct value in "answer"
        if c.offset_answer_start == 0 and c.offset_answer_end == 0:
            if c.answer != "no_answer":
                logger.error(
                    "Invalid 'no_answer': Got a prediction for position 0, but answer string is not 'no_answer'"
                )
        return c.answer == "no_answer"

    def predict_on_texts(self, question: str, texts: List[str], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a question in the supplied list of Document.
        Returns dictionaries containing answers sorted by (desc.) score.
        Example:

         ```python
         {
             'question': 'Who is the father of Arya Stark?',
             'answers':[
                          {'answer': 'Eddard,',
                          'context': " She travels with her father, Eddard, to King's Landing when he is ",
                          'offset_answer_start': 147,
                          'offset_answer_end': 154,
                          'score': 0.9787139466668613,
                          'document_id': '1337'
                          },...
                       ]
         }
         ```

        :param question: Question string
        :param documents: List of documents as string type
        :param top_k: The maximum number of answers to return
        :return: Dict containing question and answers
        """
        documents = []
        for text in texts:
            documents.append(Document(content=text))
        predictions = self.predict(question, documents, top_k)
        return predictions

    @classmethod
    def convert_to_onnx(
        cls,
        model_name: str,
        output_path: Path,
        convert_to_float16: bool = False,
        quantize: bool = False,
        task_type: str = "question_answering",
        opset_version: int = 11,
    ):
        """
        Convert a PyTorch BERT model to ONNX format and write to ./onnx-export dir. The converted ONNX model
        can be loaded with in the `FARMReader` using the export path as `model_name_or_path` param.

        Usage:

            `from haystack.reader.farm import FARMReader
            from pathlib import Path
            onnx_model_path = Path("roberta-onnx-model")
            FARMReader.convert_to_onnx(model_name="deepset/bert-base-cased-squad2", output_path=onnx_model_path)
            reader = FARMReader(onnx_model_path)`

        :param model_name: transformers model name
        :param output_path: Path to output the converted model
        :param convert_to_float16: Many models use float32 precision by default. With the half precision of float16,
                                   inference is faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs,
                                   float32 could still be be more performant.
        :param quantize: convert floating point number to integers
        :param task_type: Type of task for the model. Available options: "question_answering" or "embeddings".
        :param opset_version: ONNX opset version
        """
        AdaptiveModel.convert_to_onnx(
            model_name=model_name,
            output_path=output_path,
            task_type=task_type,
            convert_to_float16=convert_to_float16,
            quantize=quantize,
            opset_version=opset_version,
        )
