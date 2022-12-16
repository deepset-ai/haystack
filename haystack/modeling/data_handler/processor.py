# pylint: disable=missing-timeout

from typing import Optional, Dict, List, Union, Any, Iterable, Type

import os
import json
import uuid
import inspect
import logging
import random
import tarfile
import tempfile
from pathlib import Path
from inspect import signature
from abc import ABC, abstractmethod

import numpy as np
import requests
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
import transformers
from transformers import PreTrainedTokenizer, AutoTokenizer

from haystack.modeling.model.feature_extraction import (
    tokenize_batch_question_answering,
    tokenize_with_metadata,
    truncate_sequences,
)
from haystack.modeling.data_handler.dataset import convert_features_to_dataset
from haystack.modeling.data_handler.samples import (
    Sample,
    SampleBasket,
    get_passage_offsets,
    offset_to_token_idx_vecorized,
)
from haystack.modeling.data_handler.input_features import sample_to_features_text
from haystack.utils.experiment_tracking import Tracker as tracker


DOWNSTREAM_TASK_MAP = {
    "squad20": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    "covidqa": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz",
}
logger = logging.getLogger(__name__)


class Processor(ABC):
    """
    Base class for low level data processors to convert input text to PyTorch Datasets.
    """

    subclasses: dict = {}

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        train_filename: Optional[Union[Path, str]],
        dev_filename: Optional[Union[Path, str]],
        test_filename: Optional[Union[Path, str]],
        dev_split: float,
        data_dir: Optional[Union[Path, str]],
        tasks: Dict = {},
        proxies: Optional[Dict] = None,
        multithreading_rust: Optional[bool] = True,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param multithreading_rust: Whether to allow multithreading in Rust, e.g. for FastTokenizers.
                                    Note: Enabling multithreading in Rust AND multiprocessing in python might cause
                                    deadlocks.
        """
        if not multithreading_rust:
            os.environ["RAYON_RS_NUM_CPUS"] = "1"

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None  # type: ignore
        self.baskets: List = []

        self._log_params()
        self.problematic_sample_ids: set = set()

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(
        cls,
        processor_name: str,
        data_dir: str,  # TODO revert ignore
        tokenizer,  # type: ignore
        max_seq_len: int,
        train_filename: str,
        dev_filename: Optional[str],
        test_filename: str,
        dev_split: float,
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :param data_dir: Directory where data files are located.
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will sliced.
                          Only works if dev_filename is set to None
        :param kwargs: placeholder for passing generic parameters
        :return: An instance of the specified processor.
        """

        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(
            "Got more parameters than needed for loading %s: %s. Those won't be used!", processor_name, unused_args
        )
        processor = cls.subclasses[processor_name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            **kwargs,
        )

        return processor

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Infers the specific type of Processor from a config file (e.g. SquadProcessor) and loads an instance of it.

        :param load_dir: directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. SquadProcessor)
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        config["inference"] = True
        # init tokenizer
        if "lower_case" in config.keys():
            logger.warning(
                "Loading tokenizer from deprecated config. "
                "If you used `custom_vocab` or `never_split_chars`, this won't work anymore."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                load_dir, tokenizer_class=config["tokenizer"], do_lower_case=config["lower_case"]
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(load_dir, tokenizer_class=config["tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

        for task_name, task in config["tasks"].items():
            processor.add_task(
                name=task_name,
                metric=task["metric"],
                label_list=task["label_list"],
                label_column_name=task["label_column_name"],
                text_column_name=task.get("text_column_name", None),
                task_type=task["task_type"],
            )

        if processor is None:
            raise Exception

        return processor

    @classmethod
    def convert_from_transformers(
        cls,
        tokenizer_name_or_path,
        task_type,
        max_seq_len,
        doc_stride,
        revision=None,
        tokenizer_class=None,
        tokenizer_args=None,
        use_fast=True,
        **kwargs,
    ):
        tokenizer_args = tokenizer_args or {}
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            tokenizer_class=tokenizer_class,
            use_fast=use_fast,
            revision=revision,
            **tokenizer_args,
            **kwargs,
        )

        # TODO infer task_type automatically from config (if possible)
        if task_type == "question_answering":
            processor = SquadProcessor(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                label_list=["start_token", "end_token"],
                metric="squad",
                data_dir="data",
                doc_stride=doc_stride,
            )
        elif task_type == "embeddings":
            processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)

        else:
            raise ValueError(
                f"`task_type` {task_type} is not supported yet. "
                f"Valid options for arg `task_type`: 'question_answering', "
                f"'embeddings', "
            )

        return processor

    def save(self, save_dir: str):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        # self.__dict__ doesn't give parent class attributes
        for key, value in inspect.getmembers(self):
            if _is_json(value) and key[0] != "_":
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    # TODO potentially remove tasks from code - multitask learning is not supported anyways
    def add_task(
        self, name, metric, label_list, label_column_name=None, label_name=None, task_type=None, text_column_name=None
    ):
        if type(label_list) is not list:
            raise ValueError(f"Argument `label_list` must be of type list. Got: f{type(label_list)}")

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type,
        }

    @abstractmethod
    def file_to_dicts(self, file: str) -> List[dict]:
        raise NotImplementedError()

    @abstractmethod
    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    def _create_dataset(self, baskets: List[SampleBasket]):
        raise NotImplementedError

    @staticmethod
    def log_problematic(problematic_sample_ids):
        if problematic_sample_ids:
            n_problematic = len(problematic_sample_ids)
            problematic_id_str = ", ".join([str(i) for i in problematic_sample_ids])
            logger.error(
                "Unable to convert %s samples to features. Their ids are : %s", n_problematic, problematic_id_str
            )

    @staticmethod
    def _check_sample_features(basket: SampleBasket):
        """
        Check if all samples in the basket has computed its features.

        :param basket: the basket containing the samples

        :return: True if all the samples in the basket has computed its features, False otherwise
        """
        return basket.samples and not any(sample.features is None for sample in basket.samples)

    def _log_samples(self, n_samples: int, baskets: List[SampleBasket]):
        logger.debug("*** Show %s random examples ***", n_samples)
        if len(baskets) == 0:
            logger.debug("*** No samples to show because there are no baskets ***")
            return
        for i in range(n_samples):
            random_basket = random.choice(baskets)
            random_sample = random.choice(random_basket.samples)  # type: ignore
            logger.debug(random_sample)

    def _log_params(self):
        params = {"processor": self.__class__.__name__, "tokenizer": self.tokenizer.__class__.__name__}
        names = ["max_seq_len", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        tracker.track_params(params)


class SquadProcessor(Processor):
    """
    Convert QA data (in SQuAD Format)
    """

    def __init__(
        self,
        tokenizer,  # type: ignore
        max_seq_len: int,
        data_dir: Optional[Union[Path, str]],
        label_list: Optional[List[str]] = None,
        metric="squad",  # type: ignore
        train_filename: Optional[Union[Path, str]] = Path("train-v2.0.json"),
        dev_filename: Optional[Union[Path, str]] = Path("dev-v2.0.json"),
        test_filename: Optional[Union[Path, str]] = None,
        dev_split: float = 0,
        doc_stride: int = 128,
        max_query_length: int = 64,
        proxies: Optional[dict] = None,
        max_answers: int = 6,
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `haystack.basics.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/haystack/blob/main/haystack/basics/data_handler/utils.py>`_.
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :param metric: name of metric that shall be used for evaluation, can be "squad" or "top_n_accuracy"
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: None
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param doc_stride: When the document containing the answer is too long it gets split into part, strided by doc_stride
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_answers: number of answers to be converted. QA dev or train sets can contain multi-way annotations, which are converted to arrays of max_answer length
        :param kwargs: placeholder for passing generic parameters
        """
        self.ph_output_type = "per_token_squad"

        # validate max_seq_len
        assert max_seq_len <= tokenizer.model_max_length, (
            "max_seq_len cannot be greater than the maximum sequence length handled by the model: "
            f"got max_seq_len={max_seq_len}, while the model maximum length is {tokenizer.model_max_length}. "
            "Please adjust max_seq_len accordingly or use another model "
        )

        assert doc_stride < (max_seq_len - max_query_length), (
            "doc_stride ({}) is longer than max_seq_len ({}) minus space reserved for query tokens ({}). \nThis means that there will be gaps "
            "as the passage windows slide, causing the model to skip over parts of the document.\n"
            "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384)\n "
            "Or decrease max_query_length".format(doc_stride, max_seq_len, max_query_length)
        )

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_answers = max_answers
        super(SquadProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        self._initialize_special_tokens_count()
        if metric and label_list:
            self.add_task("question_answering", metric, label_list)
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for Question Answering.
        For this we have an internal representation called "baskets".
        Each basket is a question-document pair.
        Each stage adds or transforms specific information to our baskets.

        :param dicts: dict, input dictionary with SQuAD style information present
        :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: boolean, whether to return the baskets or not (baskets are needed during inference)
        """
        # Convert to standard format
        pre_baskets = [self.convert_qa_input_dict(x) for x in dicts]  # TODO move to input object conversion

        # Tokenize documents and questions
        baskets = tokenize_batch_question_answering(pre_baskets, self.tokenizer, indices)

        # Split documents into smaller passages to fit max_seq_len
        baskets = self._split_docs_into_passages(baskets)

        # Convert answers from string to token space, skip this step for inference
        if not return_baskets:
            baskets = self._convert_answers(baskets)

        # Convert internal representation (nested baskets + samples with mixed types) to pytorch features (arrays of numbers)
        baskets = self._passages_to_pytorch_features(baskets, return_baskets)

        # Convert features into pytorch dataset, this step also removes potential errors during preprocessing
        dataset, tensor_names, baskets = self._create_dataset(baskets)

        # Logging
        if indices:
            if 0 in indices:
                self._log_samples(n_samples=1, baskets=self.baskets)

        # During inference we need to keep the information contained in baskets.
        if return_baskets:
            return dataset, tensor_names, self.problematic_sample_ids, baskets
        else:
            return dataset, tensor_names, self.problematic_sample_ids

    def file_to_dicts(self, file: str) -> List[dict]:
        nested_dicts = _read_squad_file(filename=file)
        dicts = [y for x in nested_dicts for y in x["paragraphs"]]
        return dicts

    # TODO use Input Objects instead of this function, remove Natural Questions (NQ) related code
    def convert_qa_input_dict(self, infer_dict: dict) -> Dict[str, Any]:
        """Input dictionaries in QA can either have ["context", "qas"] (internal format) as keys or
        ["text", "questions"] (api format). This function converts the latter into the former. It also converts the
        is_impossible field to answer_type so that NQ and SQuAD dicts have the same format.
        """
        # validate again max_seq_len
        assert self.max_seq_len <= self.tokenizer.model_max_length, (
            "max_seq_len cannot be greater than the maximum sequence length handled by the model: "
            f"got max_seq_len={self.max_seq_len}, while the model maximum length is {self.tokenizer.model_max_length}. "
            "Please adjust max_seq_len accordingly or use another model "
        )

        # check again for doc stride vs max_seq_len when. Parameters can be changed for already initialized models (e.g. in haystack)
        assert self.doc_stride < (self.max_seq_len - self.max_query_length), (
            "doc_stride ({}) is longer than max_seq_len ({}) minus space reserved for query tokens ({}). \nThis means that there will be gaps "
            "as the passage windows slide, causing the model to skip over parts of the document.\n"
            "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384)\n "
            "Or decrease max_query_length".format(self.doc_stride, self.max_seq_len, self.max_query_length)
        )

        try:
            # Check if infer_dict is already in internal json format
            if "context" in infer_dict and "qas" in infer_dict:
                return infer_dict
            # converts dicts from inference mode to data structure used in Haystack
            questions = infer_dict["questions"]
            text = infer_dict["text"]
            uid = infer_dict.get("id", None)
            qas = [{"question": q, "id": uid, "answers": [], "answer_type": None} for i, q in enumerate(questions)]
            converted = {"qas": qas, "context": text}
            return converted
        except KeyError:
            raise Exception("Input does not have the expected format")

    def _initialize_special_tokens_count(self):
        vec = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=["a"], token_ids_1=["b"])
        self.sp_toks_start = vec.index("a")
        self.sp_toks_mid = vec.index("b") - self.sp_toks_start - 1
        self.sp_toks_end = len(vec) - vec.index("b") - 1

    def _split_docs_into_passages(self, baskets: List[SampleBasket]):
        """
        Because of the sequence length limitation of Language Models, the documents need to be divided into smaller
        parts that we call passages.
        """
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        for basket in baskets:
            samples = []
            ########## perform some basic checking
            # TODO, eventually move checking into input validation functions
            # ignore samples with empty context
            if basket.raw["document_text"] == "":
                logger.warning("Ignoring sample with empty context")
                continue
            ########## end checking

            # Calculate the number of tokens that can be reserved for the passage. This is calculated by considering
            # the max_seq_len, the number of tokens in the question and the number of special tokens that will be added
            # when the question and passage are joined (e.g. [CLS] and [SEP])
            passage_len_t = (
                self.max_seq_len - len(basket.raw["question_tokens"][: self.max_query_length]) - n_special_tokens
            )

            # passage_spans is a list of dictionaries where each defines the start and end of each passage
            # on both token and character level
            try:
                passage_spans = get_passage_offsets(
                    basket.raw["document_offsets"], self.doc_stride, passage_len_t, basket.raw["document_text"]
                )
            except Exception as e:
                logger.warning(
                    f"Could not devide document into passages. Document: {basket.raw['document_text'][:200]}\n"
                    f"With error: {e}"
                )
                passage_spans = []

            for passage_span in passage_spans:
                # Unpack each variable in the dictionary. The "_t" and "_c" indicate
                # whether the index is on the token or character level
                passage_start_t = passage_span["passage_start_t"]
                passage_end_t = passage_span["passage_end_t"]
                passage_start_c = passage_span["passage_start_c"]
                passage_end_c = passage_span["passage_end_c"]

                passage_start_of_word = basket.raw["document_start_of_word"][passage_start_t:passage_end_t]
                passage_tokens = basket.raw["document_tokens"][passage_start_t:passage_end_t]
                passage_text = basket.raw["document_text"][passage_start_c:passage_end_c]

                clear_text = {
                    "passage_text": passage_text,
                    "question_text": basket.raw["question_text"],
                    "passage_id": passage_span["passage_id"],
                }
                tokenized = {
                    "passage_start_t": passage_start_t,
                    "passage_start_c": passage_start_c,
                    "passage_tokens": passage_tokens,
                    "passage_start_of_word": passage_start_of_word,
                    "question_tokens": basket.raw["question_tokens"][: self.max_query_length],
                    "question_offsets": basket.raw["question_offsets"][: self.max_query_length],
                    "question_start_of_word": basket.raw["question_start_of_word"][: self.max_query_length],
                }
                # The sample ID consists of internal_id and a passage numbering
                sample_id = f"{basket.id_internal}-{passage_span['passage_id']}"
                samples.append(Sample(id=sample_id, clear_text=clear_text, tokenized=tokenized))

            basket.samples = samples

        return baskets

    def _convert_answers(self, baskets: List[SampleBasket]):
        """
        Converts answers that are pure strings into the token based representation with start and end token offset.
        Can handle multiple answers per question document pair as is common for development/text sets
        """
        for basket in baskets:
            error_in_answer = False
            for num, sample in enumerate(basket.samples):  # type: ignore
                # Dealing with potentially multiple answers (e.g. Squad dev set)
                # Initializing a numpy array of shape (max_answers, 2), filled with -1 for missing values
                label_idxs = np.full((self.max_answers, 2), fill_value=-1)

                if error_in_answer or (len(basket.raw["answers"]) == 0):
                    # If there are no answers we set
                    label_idxs[0, :] = 0
                else:
                    # For all other cases we use start and end token indices, that are relative to the passage
                    for i, answer in enumerate(basket.raw["answers"]):
                        # Calculate start and end relative to document
                        answer_len_c = len(answer["text"])
                        answer_start_c = answer["answer_start"]
                        answer_end_c = answer_start_c + answer_len_c - 1

                        # Convert character offsets to token offsets on document level
                        answer_start_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_start_c)
                        answer_end_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_end_c)

                        # Adjust token offsets to be relative to the passage
                        answer_start_t -= sample.tokenized["passage_start_t"]  # type: ignore
                        answer_end_t -= sample.tokenized["passage_start_t"]  # type: ignore

                        # Initialize some basic variables
                        question_len_t = len(sample.tokenized["question_tokens"])  # type: ignore
                        passage_len_t = len(sample.tokenized["passage_tokens"])  # type: ignore

                        # Check that start and end are contained within this passage
                        # answer_end_t is 0 if the first token is the answer
                        # answer_end_t is passage_len_t if the last token is the answer
                        if passage_len_t > answer_start_t >= 0 and passage_len_t >= answer_end_t >= 0:
                            # Then adjust the start and end offsets by adding question and special token
                            label_idxs[i][0] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_start_t
                            label_idxs[i][1] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_end_t
                        # If the start or end of the span answer is outside the passage, treat passage as no_answer
                        else:
                            label_idxs[i][0] = 0
                            label_idxs[i][1] = 0

                        ########## answer checking ##############################
                        # TODO, move this checking into input validation functions and delete wrong examples there
                        # Cases where the answer is not within the current passage will be turned into no answers by the featurization fn
                        if answer_start_t < 0 or answer_end_t >= passage_len_t:
                            pass
                        else:
                            doc_text = basket.raw["document_text"]
                            answer_indices = doc_text[answer_start_c : answer_end_c + 1]
                            answer_text = answer["text"]
                            # check if answer string can be found in context
                            if answer_text not in doc_text:
                                logger.warning(
                                    f"Answer '{answer['text']}' not contained in context.\n"
                                    f"Example will not be converted for training/evaluation."
                                )
                                error_in_answer = True
                                label_idxs[i][0] = -100  # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break  # Break loop around answers, so the error message is not shown multiple times
                            if answer_indices.strip() != answer_text.strip():
                                logger.warning(
                                    f"Answer using start/end indices is '{answer_indices}' while gold label text is '{answer_text}'.\n"
                                    f"Example will not be converted for training/evaluation."
                                )
                                error_in_answer = True
                                label_idxs[i][0] = -100  # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break  # Break loop around answers, so the error message is not shown multiple times
                        ########## end of checking ####################

                sample.tokenized["labels"] = label_idxs  # type: ignore

        return baskets

    def _passages_to_pytorch_features(self, baskets: List[SampleBasket], return_baskets: bool):
        """
        Convert internal representation (nested baskets + samples with mixed types) to python features (arrays of numbers).
        We first join question and passages into one large vector.
        Then we add vectors for: - input_ids (token ids)
                                 - segment_ids (does a token belong to question or document)
                                 - padding_mask
                                 - span_mask (valid answer tokens)
                                 - start_of_word
        """
        for basket in baskets:
            # Add features to samples
            for num, sample in enumerate(basket.samples):  # type: ignore
                # Initialize some basic variables
                if sample.tokenized is not None:
                    question_tokens = sample.tokenized["question_tokens"]
                    question_start_of_word = sample.tokenized["question_start_of_word"]
                    question_len_t = len(question_tokens)
                    passage_start_t = sample.tokenized["passage_start_t"]
                    passage_tokens = sample.tokenized["passage_tokens"]
                    passage_start_of_word = sample.tokenized["passage_start_of_word"]
                    passage_len_t = len(passage_tokens)
                    sample_id = [int(x) for x in sample.id.split("-")]

                    # - Combines question_tokens and passage_tokens into a single vector called input_ids
                    # - input_ids also contains special tokens (e.g. CLS or SEP tokens).
                    # - It will have length = question_len_t + passage_len_t + n_special_tokens. This may be less than
                    #   max_seq_len but never greater since truncation was already performed when the document was chunked into passages
                    question_input_ids = sample.tokenized["question_tokens"]
                    passage_input_ids = sample.tokenized["passage_tokens"]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(
                    token_ids_0=question_input_ids, token_ids_1=passage_input_ids
                )

                segment_ids = self.tokenizer.create_token_type_ids_from_sequences(
                    token_ids_0=question_input_ids, token_ids_1=passage_input_ids
                )
                # To make the start index of passage tokens the start manually
                seq_2_start_t = self.sp_toks_start + question_len_t + self.sp_toks_mid

                start_of_word = (
                    [0] * self.sp_toks_start
                    + question_start_of_word
                    + [0] * self.sp_toks_mid
                    + passage_start_of_word
                    + [0] * self.sp_toks_end
                )

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                padding_mask = [1] * len(input_ids)

                # The span_mask has 1 for tokens that are valid start or end tokens for QA spans.
                # 0s are assigned to question tokens, mid special tokens, end special tokens, and padding
                # Note that start special tokens are assigned 1 since they can be chosen for a no_answer prediction
                span_mask = [1] * self.sp_toks_start
                span_mask += [0] * question_len_t
                span_mask += [0] * self.sp_toks_mid
                span_mask += [1] * passage_len_t
                span_mask += [0] * self.sp_toks_end

                # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
                pad_idx = self.tokenizer.pad_token_id
                padding = [pad_idx] * (self.max_seq_len - len(input_ids))
                zero_padding = [0] * (self.max_seq_len - len(input_ids))

                input_ids += padding
                padding_mask += zero_padding
                segment_ids += zero_padding
                start_of_word += zero_padding
                span_mask += zero_padding

                # TODO possibly remove these checks after input validation is in place
                len_check = (
                    len(input_ids) == len(padding_mask) == len(segment_ids) == len(start_of_word) == len(span_mask)
                )
                id_check = len(sample_id) == 3
                label_check = return_baskets or len(sample.tokenized.get("labels", [])) == self.max_answers  # type: ignore
                # labels are set to -100 when answer cannot be found
                label_check2 = return_baskets or np.all(sample.tokenized["labels"] > -99)  # type: ignore
                if len_check and id_check and label_check and label_check2:
                    # - The first of the labels will be used in train, and the full array will be used in eval.
                    # - start_of_word and spec_tok_mask are not actually needed by model.forward() but are needed for
                    #   model.formatted_preds() during inference for creating answer strings
                    # - passage_start_t is index of passage's first token relative to document
                    feature_dict = {
                        "input_ids": input_ids,
                        "padding_mask": padding_mask,
                        "segment_ids": segment_ids,
                        "passage_start_t": passage_start_t,
                        "start_of_word": start_of_word,
                        "labels": sample.tokenized.get("labels", []),  # type: ignore
                        "id": sample_id,
                        "seq_2_start_t": seq_2_start_t,
                        "span_mask": span_mask,
                    }
                    # other processor's features can be lists
                    sample.features = [feature_dict]  # type: ignore
                else:
                    self.problematic_sample_ids.add(sample.id)
                    sample.features = None
        return baskets

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List[dict] = []
        basket_to_remove = []
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  # type: ignore
                    features_flat.extend(sample.features)  # type: ignore
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, baskets


class TextSimilarityProcessor(Processor):
    """
    Used to handle the Dense Passage Retrieval (DPR) datasets that come in json format, example: biencoder-nq-train.json, biencoder-nq-dev.json, trivia-train.json, trivia-dev.json

    Datasets can be downloaded from the official DPR github repository (https://github.com/facebookresearch/DPR)
    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {"dataset": str,
    "question": str,
    "answers": list of str
    "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    """

    def __init__(
        self,
        query_tokenizer,  # type: ignore
        passage_tokenizer,  # type: ignore
        max_seq_len_query: int,
        max_seq_len_passage: int,
        data_dir: str = "",
        metric=None,  # type: ignore
        train_filename: str = "train.json",
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = "test.json",
        dev_split: float = 0.1,
        proxies: Optional[dict] = None,
        max_samples: Optional[int] = None,
        embed_title: bool = True,
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `haystack.basics.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/haystack/blob/main/haystack/basics/data_handler/utils.py>`_.
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: None
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :param kwargs: placeholder for passing generic parameters
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = query_tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.embed_title = embed_title
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage

        super(TextSimilarityProcessor, self).__init__(
            tokenizer=None,  # type: ignore
            max_seq_len=0,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(
                name="text_similarity",
                metric=metric,
                label_list=label_list,
                label_name="label",
                task_type="text_similarity",
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Overwriting method from parent class to **always** load the TextSimilarityProcessor instead of the specific class stored in the config.

        :param load_dir: directory that contains a 'processor_config.json'
        :return: An instance of an TextSimilarityProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizers
        query_tokenizer_class: Type[PreTrainedTokenizer] = getattr(transformers, config["query_tokenizer"])
        query_tokenizer = query_tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=load_dir, subfolder="query"
        )
        passage_tokenizer_class: Type[PreTrainedTokenizer] = getattr(transformers, config["passage_tokenizer"])
        passage_tokenizer = passage_tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=load_dir, subfolder="passage"
        )

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["query_tokenizer"]
        del config["passage_tokenizer"]

        processor = cls.load(
            query_tokenizer=query_tokenizer,
            passage_tokenizer=passage_tokenizer,
            processor_name="TextSimilarityProcessor",
            **config,
        )
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def save(self, save_dir: Union[str, Path]):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["query_tokenizer"] = self.query_tokenizer.__class__.__name__
        config["passage_tokenizer"] = self.passage_tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.query_tokenizer.save_pretrained(str(save_dir / "query"))
        self.passage_tokenizer.save_pretrained(str(save_dir / "passage"))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for TextSimilarity (e.g. DPR).
        For conversion we have an internal representation called "baskets".
        Each basket is one query and related text passages (positive passages fitting to the query and negative
        passages that do not fit the query)
        Each stage adds or transforms specific information to our baskets.

        :param dicts: input dictionary with DPR-style content
                        {"query": str,
                         "passages": List[
                                        {'title': str,
                                        'text': str,
                                        'label': 'hard_negative',
                                        'external_id': str},
                                        ....
                                        ]
                         }
        :param indices: indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: whether to return the baskets or not (baskets are needed during inference)
        :return: dataset, tensor_names, problematic_ids, [baskets]
        """
        # Take the dict and insert into our basket structure, this stages also adds an internal IDs
        baskets = self._fill_baskets(dicts, indices)

        # Separat conversion of query
        baskets = self._convert_queries(baskets=baskets)

        # and context passages. When converting the context the label is also assigned.
        baskets = self._convert_contexts(baskets=baskets)

        # Convert features into pytorch dataset, this step also removes and logs potential errors during preprocessing
        dataset, tensor_names, problematic_ids, baskets = self._create_dataset(baskets)

        if problematic_ids:
            logger.error(
                f"There were {len(problematic_ids)} errors during preprocessing at positions: {problematic_ids}"
            )

        if return_baskets:
            return dataset, tensor_names, problematic_ids, baskets
        else:
            return dataset, tensor_names, problematic_ids

    def file_to_dicts(self, file: str) -> List[dict]:
        """
        Converts a Dense Passage Retrieval (DPR) data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format
                Each sample is a dictionary of format:
                {"dataset": str,
                "question": str,
                "answers": list of str
                "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                }


        Returns:
        list of dictionaries: List[dict]
            each dictionary:
            {"query": str,
            "passages": [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
            {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
            ...]}
        """
        dicts = _read_dpr_json(
            file,
            max_samples=self.max_samples,
            num_hard_negatives=self.num_hard_negatives,
            num_positives=self.num_positives,
            shuffle_negatives=self.shuffle_negatives,
            shuffle_positives=self.shuffle_positives,
        )

        # shuffle dicts to make sure that similar positive passages do not end up in one batch
        dicts = random.sample(dicts, len(dicts))
        return dicts

    def _fill_baskets(self, dicts: List[dict], indices: Optional[List[int]]):
        baskets = []
        if not indices:
            indices = list(range(len(dicts)))
        for d, id_internal in zip(dicts, indices):
            basket = SampleBasket(id_external=None, id_internal=id_internal, raw=d)
            baskets.append(basket)
        return baskets

    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query = self._normalize_question(basket.raw["query"])

                    # featurize the query
                    query_inputs = self.query_tokenizer(
                        query,
                        max_length=self.max_seq_len_query,
                        add_special_tokens=True,
                        truncation=True,
                        truncation_strategy="longest_first",
                        padding="max_length",
                        return_token_type_ids=True,
                    )

                    # tokenize query
                    tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize"
                        )
                        return None

                    clear_text["query_text"] = query
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_inputs["input_ids"]
                    features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                    features[0]["query_attention_mask"] = query_inputs["attention_mask"]
                except Exception as e:
                    features = None  # type: ignore

            sample = Sample(id="", clear_text=clear_text, tokenized=tokenized, features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def _convert_contexts(self, baskets: List[SampleBasket]):
        for basket in baskets:
            if "passages" in basket.raw:
                try:
                    positive_context = list(filter(lambda x: x["label"] == "positive", basket.raw["passages"]))
                    if self.shuffle_positives:
                        random.shuffle(positive_context)
                    positive_context = positive_context[: self.num_positives]
                    hard_negative_context = list(
                        filter(lambda x: x["label"] == "hard_negative", basket.raw["passages"])
                    )
                    if self.shuffle_negatives:
                        random.shuffle(hard_negative_context)
                    hard_negative_context = hard_negative_context[: self.num_hard_negatives]

                    positive_ctx_titles = [passage.get("title", None) for passage in positive_context]
                    positive_ctx_texts = [passage["text"] for passage in positive_context]
                    hard_negative_ctx_titles = [passage.get("title", None) for passage in hard_negative_context]
                    hard_negative_ctx_texts = [passage["text"] for passage in hard_negative_context]

                    # all context passages and labels: 1 for positive context and 0 for hard-negative context
                    ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
                    # featurize context passages
                    if self.embed_title:
                        # concatenate title with positive context passages + negative context passages
                        all_ctx = self._combine_title_context(
                            positive_ctx_titles, positive_ctx_texts
                        ) + self._combine_title_context(hard_negative_ctx_titles, hard_negative_ctx_texts)
                    else:
                        all_ctx = positive_ctx_texts + hard_negative_ctx_texts

                    # assign empty string tuples if hard_negative passages less than num_hard_negatives
                    all_ctx += [("", "")] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

                    ctx_inputs = self.passage_tokenizer(
                        all_ctx,
                        add_special_tokens=True,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_seq_len_passage,
                        return_token_type_ids=True,
                    )

                    ctx_segment_ids = [[0] * len(ctx_inputs["token_type_ids"][0])] * len(ctx_inputs["token_type_ids"])

                    # get tokens in string format
                    tokenized_passage = [
                        self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in ctx_inputs["input_ids"]
                    ]

                    # for DPR we only have one sample containing query and corresponding (multiple) context features
                    sample = basket.samples[0]  # type: ignore
                    sample.clear_text["passages"] = positive_context + hard_negative_context
                    sample.tokenized["passages_tokens"] = tokenized_passage  # type: ignore
                    sample.features[0]["passage_input_ids"] = ctx_inputs["input_ids"]  # type: ignore
                    sample.features[0]["passage_segment_ids"] = ctx_segment_ids  # type: ignore
                    sample.features[0]["passage_attention_mask"] = ctx_inputs["attention_mask"]  # type: ignore
                    sample.features[0]["label_ids"] = ctx_label  # type: ignore
                except Exception as e:
                    basket.samples[0].features = None  # type: ignore

        return baskets

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List[dict] = []
        basket_to_remove = []
        problematic_ids: set = set()
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  # type: ignore
                    features_flat.extend(sample.features)  # type: ignore
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Removes '?' from queries/questions"""
        if question[-1] == "?":
            question = question[:-1]
        return question

    @staticmethod
    def _combine_title_context(titles: List[str], texts: List[str]):
        res = []
        for title, ctx in zip(titles, texts):
            if title is None:
                title = ""
                logger.warning(
                    f"Couldn't find title although `embed_title` is set to True for DPR. Using title='' now. Related passage text: '{ctx}' "
                )
            res.append(tuple((title, ctx)))
        return res


class TableTextSimilarityProcessor(Processor):
    """
    Used to handle the Multimodal Retrieval datasets consisting of text passages and tables
    that come in json format.

    """

    def __init__(
        self,
        query_tokenizer,  # type: ignore
        passage_tokenizer,  # type: ignore
        table_tokenizer,  # type: ignore
        max_seq_len_query: int,
        max_seq_len_passage: int,
        max_seq_len_table: int,
        data_dir: str = "",
        metric: Optional[str] = None,
        train_filename: Optional[Union[Path, str]] = "train.json",
        dev_filename: Optional[Union[Path, str]] = None,
        test_filename: Optional[Union[Path, str]] = "test.json",
        dev_split: float = 0.1,
        proxies: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        embed_meta_fields: List[str] = ["page_title", "section_title", "caption"],
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a text passage (str) into tokens.
        :param table_tokenizer: Used to split a table into tokens
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param max_seq_len_table: Table samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automatically
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict DOWNSTREAM_TASK_MAP
        :param metric: Name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing the test data.
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None.
        :param proxies: Proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use.
        :param embed_meta_fields: List of meta fields to embed in text passages and tables during tensorization.
        :param num_hard_negatives: Maximum number of hard negative context passages in a sample.
        :param num_positives: Maximum number of positive context passages in a sample.
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the
                                  num_hard_negative number of passages.
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the
                                  num_positive number of passages.
        :param label_list: List of labels to predict. Usually ["hard_negative", "positive"].
        :param kwargs: Placeholder for passing generic parameters
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = query_tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.table_tokenizer = table_tokenizer
        self.embed_meta_fields = embed_meta_fields
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage
        self.max_seq_len_table = max_seq_len_table

        super(TableTextSimilarityProcessor, self).__init__(
            tokenizer=self.query_tokenizer,
            max_seq_len=0,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(
                name="text_similarity",
                metric=metric,
                label_list=label_list,
                label_name="label",
                task_type="text_similarity",
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Overwriting method from parent class to **always** load the TableTextSimilarityProcessor
         instead of the specific class stored in the config.

        :param load_dir: Directory that contains a 'processor_config.json'
        :return: An instance of an TableTextSimilarityProcessor.
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizer
        query_tokenizer = AutoTokenizer.from_pretrained(
            load_dir, tokenizer_class=config["query_tokenizer"], subfolder="query"
        )
        passage_tokenizer = AutoTokenizer.from_pretrained(
            load_dir, tokenizer_class=config["passage_tokenizer"], subfolder="passage"
        )
        table_tokenizer = AutoTokenizer.from_pretrained(
            load_dir, tokenizer_class=config["table_tokenizer"], subfolder="table"
        )

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["query_tokenizer"]
        del config["passage_tokenizer"]
        del config["table_tokenizer"]

        processor = cls.load(
            query_tokenizer=query_tokenizer,
            passage_tokenizer=passage_tokenizer,
            table_tokenizer=table_tokenizer,
            processor_name="TableTextSimilarityProcessor",
            **config,
        )
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def save(self, save_dir: Union[str, Path]):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["query_tokenizer"] = self.query_tokenizer.__class__.__name__
        config["passage_tokenizer"] = self.passage_tokenizer.__class__.__name__
        config["table_tokenizer"] = self.table_tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.query_tokenizer.save_pretrained(str(save_dir / "query"))
        self.passage_tokenizer.save_pretrained(str(save_dir / "passage"))
        self.table_tokenizer.save_pretrained(str(save_dir / "table"))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def file_to_dicts(self, file: str) -> List[Dict]:
        """
        Converts a Multimodal Retrieval data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format
                Each sample is a dictionary of format:
                {"question": str,
                "answers": list of str
                "positive_ctxs": list of dictionaries of format
                    {'title': str, 'text': str, 'passage_id': str, 'type': 'text', 'source': str}
                    or
                    {'page_title': str, 'section_title': str, 'caption': str, 'columns': list of str,
                     'rows': list of list of str, 'type': 'table', 'source': str}
                "hard_negative_ctxs": list of dictionaries of format
                    {'title': str, 'text': str, 'passage_id': str, 'type': 'text', 'source': str}
                    or
                    {'page_title': str, 'section_title': str, 'caption': str, 'columns': list of str,
                     'rows': list of list of str, 'type': 'table', 'source': str}
                }


        Returns:
        List of dictionaries: List[dict]
            each dictionary:
            {"query": str,
            "passages": [
                {"title": str, "text": str, "label": "positive" / "hard_negative", "type": "text", "external_id": id}
                or
                {"page_title": str, "section_title": str, "caption": str, "columns": list of str,
                 "rows": list of list of str, "label": "positive" / "hard_negative", "type": "table", "external_id": id}
            ...]}
        """
        dicts = self._read_multimodal_dpr_json(file, max_samples=self.max_samples)
        return dicts

    def _read_multimodal_dpr_json(self, file: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Reads a Multimodal Retrieval data file in json format and returns a list of dictionaries.

        :param file: filename of MMR data in json format

        Returns:
            list of dictionaries: List[dict]
            each dictionary: {
                        "query": str -> query_text
                        "passages": List[dictionaries] -> [
                                    {"text": str, "title": str, "label": "positive" / "hard_negative, "external_id": id},
                                    or
                                    {"page_title": str, "section_title": str, "caption": str, "columns": list of str,
                                     "rows": list of lists of str, "label": "positive" / "hard_negative", "type": "table", "external_id": id}
                                    ...]
                        }
        """
        dicts = json.load(open(file))
        if max_samples:
            dicts = random.sample(dicts, min(max_samples, len(dicts)))
        # convert DPR dictionary to standard dictionary
        query_json_keys = ["question", "questions", "query"]
        positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
        hard_negative_json_keys = [
            "hard_negative_contexts",
            "hard_negative_ctxs",
            "hard_negative_context",
            "hard_negative_ctx",
        ]
        standard_dicts = []
        for dict in dicts:
            sample = {}
            docs = []
            for key, val in dict.items():
                if key in query_json_keys:
                    sample["query"] = val
                elif key in positive_context_json_keys + hard_negative_json_keys:
                    for doc in val:
                        if doc["type"] == "table":
                            docs.append(
                                {
                                    "meta": [
                                        doc[meta_field] for meta_field in self.embed_meta_fields if meta_field in doc
                                    ],
                                    "columns": doc.get("columns"),
                                    "rows": doc.get("rows"),
                                    "label": "positive" if key in positive_context_json_keys else "hard_negative",
                                    "type": "table",
                                }
                            )
                        elif doc["type"] == "text":
                            docs.append(
                                {
                                    "meta": [
                                        doc[meta_field] for meta_field in self.embed_meta_fields if meta_field in doc
                                    ],
                                    "text": doc["text"],
                                    "label": "positive" if key in positive_context_json_keys else "hard_negative",
                                    "type": "text",
                                }
                            )

                sample["passages"] = docs
            standard_dicts.append(sample)
        return standard_dicts

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for TextSimilarity.
        For conversion we have an internal representation called "baskets".
        Each basket is one query and related text passages (positive passages fitting to the query and negative
        passages that do not fit the query)
        Each stage adds or transforms specific information to our baskets.

        :param dicts: List of dicts, input dictionary with DPR-style content
                        {"query": str,
                         "passages": List[
                                        {'title': str,
                                        'text': str,
                                        'label': 'hard_negative',
                                        'external_id': str},
                                        ....
                                        ]
                         }
        :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: boolean, whether to return the baskets or not (baskets are needed during inference)
        """

        # Take the dict and insert into our basket structure, this stages also adds an internal IDs
        baskets = self._fill_baskets(dicts, indices)

        # Separate conversion of query
        baskets = self._convert_queries(baskets=baskets)

        # and context passages and tables. When converting the context the label is also assigned.
        baskets = self._convert_contexts(baskets=baskets)

        # Convert features into pytorch dataset, this step also removes and logs potential errors during preprocessing
        dataset, tensor_names, problematic_ids, baskets = self._create_dataset(baskets)

        if problematic_ids:
            logger.error(
                f"There were {len(problematic_ids)} errors during preprocessing at positions: {problematic_ids}"
            )

        if return_baskets:
            return dataset, tensor_names, problematic_ids, baskets
        else:
            return dataset, tensor_names, problematic_ids

    def _fill_baskets(self, dicts: List[Dict], indices: Optional[Iterable[int]]):
        baskets = []
        if not indices:
            indices = range(len(dicts))
        for d, id_internal in zip(dicts, indices):
            basket = SampleBasket(id_external=None, id_internal=id_internal, raw=d)
            baskets.append(basket)
        return baskets

    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features: List[Dict] = [{}]
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query = self._normalize_question(basket.raw["query"])

                    # featurize the query
                    query_inputs = self.query_tokenizer(
                        query,
                        max_length=self.max_seq_len_query,
                        add_special_tokens=True,
                        truncation=True,
                        truncation_strategy="longest_first",
                        padding="max_length",
                        return_token_type_ids=True,
                    )

                    # tokenize query
                    tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize"
                        )
                        return None

                    clear_text["query_text"] = query
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_inputs["input_ids"]
                    features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                    features[0]["query_attention_mask"] = query_inputs["attention_mask"]
                except Exception as e:
                    features = None  # type: ignore

            sample = Sample(id="", clear_text=clear_text, tokenized=tokenized, features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def _convert_contexts(self, baskets: List[SampleBasket]):
        # Converts both text passages and tables.

        for basket in baskets:
            if "passages" in basket.raw:
                try:
                    positive_context = list(filter(lambda x: x["label"] == "positive", basket.raw["passages"]))
                    if self.shuffle_positives:
                        random.shuffle(positive_context)
                    positive_context = positive_context[: self.num_positives]
                    hard_negative_context = list(
                        filter(lambda x: x["label"] == "hard_negative", basket.raw["passages"])
                    )
                    if self.shuffle_negatives:
                        random.shuffle(hard_negative_context)
                    hard_negative_context = hard_negative_context[: self.num_hard_negatives]

                    positive_ctx_meta = []
                    positive_ctx_texts = []
                    hard_negative_ctx_meta = []
                    hard_negative_ctx_texts = []
                    is_table = []

                    for pos_ctx in positive_context:
                        if pos_ctx["type"] == "text":
                            positive_ctx_meta.append(" ".join(pos_ctx.get("meta")))
                            positive_ctx_texts.append(pos_ctx["text"])
                            is_table.append(0)
                        elif pos_ctx["type"] == "table":
                            positive_ctx_meta.append(" ".join(pos_ctx.get("meta")))
                            linearized_rows = [cell for row in pos_ctx["rows"] for cell in row]
                            linearized_table = " ".join(pos_ctx["columns"]) + " " + " ".join(linearized_rows)
                            positive_ctx_texts.append(linearized_table)
                            is_table.append(1)

                    for hn_ctx in hard_negative_context:
                        if hn_ctx["type"] == "text":
                            hard_negative_ctx_meta.append(" ".join(hn_ctx.get("meta")))
                            hard_negative_ctx_texts.append(hn_ctx["text"])
                            is_table.append(0)
                        elif hn_ctx["type"] == "table":
                            hard_negative_ctx_meta.append(" ".join(hn_ctx.get("meta")))
                            linearized_rows = [cell for row in hn_ctx["rows"] for cell in row]
                            linearized_table = " ".join(hn_ctx["columns"]) + " " + " ".join(linearized_rows)
                            hard_negative_ctx_texts.append(linearized_table)
                            is_table.append(1)

                    # all context passages and labels: 1 for positive context and 0 for hard-negative context
                    ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
                    # featurize context passages
                    if self.embed_meta_fields:
                        # concatenate title with positive context passages + negative context passages
                        all_ctx = self._combine_meta_context(
                            positive_ctx_meta, positive_ctx_texts
                        ) + self._combine_meta_context(hard_negative_ctx_meta, hard_negative_ctx_texts)
                    else:
                        all_ctx = positive_ctx_texts + hard_negative_ctx_texts

                    # assign empty string tuples if hard_negative passages less than num_hard_negatives
                    all_ctx += [("", "")] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

                    inputs = self.passage_tokenizer(
                        all_ctx,
                        add_special_tokens=True,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_seq_len_passage,
                        return_token_type_ids=True,
                    )

                    input_ids = inputs["input_ids"]
                    passage_segment_ids = inputs["token_type_ids"]
                    attention_mask = inputs["attention_mask"]

                    # get tokens in string format
                    tokenized = [self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in input_ids]

                    # for DPR we only have one sample containing query and corresponding (multiple) context features
                    sample = basket.samples[0]  # type: ignore
                    sample.clear_text["passages"] = positive_context + hard_negative_context  # type: ignore
                    sample.tokenized["passages_tokens"] = tokenized  # type: ignore
                    sample.features[0]["passage_input_ids"] = input_ids  # type: ignore
                    sample.features[0]["passage_segment_ids"] = passage_segment_ids  # type: ignore
                    sample.features[0]["table_segment_ids"] = passage_segment_ids  # type: ignore
                    sample.features[0]["passage_attention_mask"] = attention_mask  # type: ignore
                    sample.features[0]["label_ids"] = ctx_label  # type: ignore
                    sample.features[0]["is_table"] = is_table  # type: ignore
                except Exception as e:
                    basket.samples[0].features = None  # type: ignore

        return baskets

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List = []
        basket_to_remove = []
        problematic_ids = set()
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  # type: ignore
                    features_flat.extend(sample.features)  # type: ignore
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Removes '?' from queries/questions"""
        if question[-1] == "?":
            question = question[:-1]
        return question

    @staticmethod
    def _combine_meta_context(meta_fields: List[str], texts: List[str]):
        res = []
        for meta, ctx in zip(meta_fields, texts):
            if meta is None:
                meta = ""
            res.append(tuple((meta, ctx)))
        return res


class TextClassificationProcessor(Processor):
    """
    Used to handle the text classification datasets that come in tabular format (CSV, TSV, etc.)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        dev_stratification=False,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/main/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param dev_stratification: if True, create a class-stratified split for the dev set.
        :type dev_stratification: bool
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param multilabel: set to True for multilabel classification
        :type multilabel: bool
        :param header: which line to use as a header in the input csv/tsv
        :type  header: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples
        self.dev_stratification = dev_stratification
        logger.debug("Currently no support in Processor for returning problematic ids")

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(
                name="text_classification",
                metric=metric,
                label_list=label_list,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                task_type=task_type,
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def file_to_dicts(self, file: str) -> List[Dict]:
        raise NotImplementedError

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        self.baskets = []
        # Tokenize in batches
        texts = [x["text"] for x in dicts]
        tokenized_batch = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )
        input_ids_batch = tokenized_batch["input_ids"]
        segment_ids_batch = tokenized_batch["token_type_ids"]
        padding_masks_batch = tokenized_batch["attention_mask"]
        tokens_batch = [x.tokens for x in tokenized_batch.encodings]

        # From here we operate on a per sample basis
        for dictionary, input_ids, segment_ids, padding_mask, tokens in zip(
            dicts, input_ids_batch, segment_ids_batch, padding_masks_batch, tokens_batch
        ):

            tokenized = {}
            if debug:
                tokenized["tokens"] = tokens

            feat_dict = {"input_ids": input_ids, "padding_mask": padding_mask, "segment_ids": segment_ids}

            # Create labels
            # i.e. not inference
            if not return_baskets:
                label_dict = self.convert_labels(dictionary)
                feat_dict.update(label_dict)

            # Add Basket to self.baskets
            curr_sample = Sample(id="", clear_text=dictionary, tokenized=tokenized, features=[feat_dict])
            curr_basket = SampleBasket(id_internal=None, raw=dictionary, id_external=None, samples=[curr_sample])
            self.baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(n_samples=1, baskets=self.baskets)

        # TODO populate problematic ids
        problematic_ids: set = set()
        dataset, tensornames = self._create_dataset()
        if return_baskets:
            return dataset, tensornames, problematic_ids, self.baskets
        else:
            return dataset, tensornames, problematic_ids

    def convert_labels(self, dictionary: Dict):
        ret: Dict = {}
        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            label_name = task["label_name"]
            label_raw = dictionary[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                label_ids = [label_list.index(label_raw)]
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            ret[task["label_tensor_name"]] = label_ids
        return ret

    def _create_dataset(self):
        # TODO this is the proposed new version to replace the mother function
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names


class InferenceProcessor(TextClassificationProcessor):
    """
    Generic processor used at inference time:
    - fast
    - no labels
    - pure encoding of text into pytorch dataset
    - Doesn't read from file, but only consumes dictionaries (e.g. coming from API requests)
    """

    def __init__(self, tokenizer, max_seq_len, **kwargs):

        super(InferenceProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=None,
            dev_filename=None,
            test_filename=None,
            dev_split=None,
            data_dir=None,
            tasks={},
        )

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Overwriting method from parent class to **always** load the InferenceProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an InferenceProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_dir, tokenizer_class=config["tokenizer"])
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name="InferenceProcessor", **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def file_to_dicts(self, file: str) -> List[Dict]:
        raise NotImplementedError

    def convert_labels(self, dictionary: Dict):
        # For inference we do not need labels
        ret: Dict = {}
        return ret

    # Private method to keep s3e pooling and embedding extraction working
    def _dict_to_samples(self, dictionary: Dict, **kwargs) -> Sample:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        truncated_tokens = {}
        for seq_name, tokens in tokenized.items():
            truncated_tokens[seq_name], _, _ = truncate_sequences(
                seq_a=tokens, seq_b=None, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len
            )
        return Sample(id="", clear_text=dictionary, tokenized=truncated_tokens)

    # Private method to keep s3e pooling and embedding extraction working
    def _sample_to_features(self, sample: Sample) -> Dict:
        features = sample_to_features_text(
            sample=sample, tasks=self.tasks, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer
        )
        return features


class UnlabeledTextProcessor(Processor):
    """
    Processor to be used for distilling a teacher model into a student model from scratch. Can only be used with distil_intermediate_layers_from.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        train_filename: Optional[Union[Path, str]] = None,
        dev_filename: Optional[Union[Path, str]] = None,
        test_filename: Optional[Union[Path, str]] = None,
        dev_split: float = 0,
        data_dir: Optional[Union[Path, str]] = None,
        tasks: Dict = {},
        proxies: Optional[Dict] = None,
        multithreading_rust: Optional[bool] = True,
    ):
        super().__init__(
            tokenizer,
            max_seq_len,
            train_filename,
            dev_filename,
            test_filename,
            dev_split,
            data_dir,
            tasks,
            proxies,
            multithreading_rust,
        )
        self.add_task("question_answering", "squad", ["start_token", "end_token"])

    def file_to_dicts(self, file: str) -> List[dict]:
        dicts = []
        with open(file, "r") as f:
            for line in f:
                dicts.append({"text": line})
        return dicts

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        if return_baskets:
            raise NotImplementedError("return_baskets is not supported by UnlabeledTextProcessor")
        texts = [dict_["text"] for dict_ in dicts]
        tokens = self.tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        names = [key for key in tokens]
        inputs = [tokens[key] for key in tokens]
        if not "padding_mask" in names:
            index = names.index("attention_mask")
            names[index] = "padding_mask"
        if not "segment_ids" in names:
            index = names.index("token_type_ids")
            names[index] = "segment_ids"

        dataset = TensorDataset(*inputs)
        return dataset, names, []

    def _create_dataset(self, baskets: List[SampleBasket]):
        raise NotImplementedError("_create_dataset is not supported by UnlabeledTextProcessor")


# helper fcts
def write_squad_predictions(predictions, out_filename, predictions_filename=None):
    predictions_json = {}
    for x in predictions:
        for p in x["predictions"]:
            if p["answers"][0]["answer"] is not None:
                predictions_json[p["question_id"]] = p["answers"][0]["answer"]
            else:
                predictions_json[
                    p["question_id"]
                ] = ""  # convert No answer = None to format understood by the SQuAD eval script

    if predictions_filename:
        dev_labels = {}
        temp = json.load(open(predictions_filename, "r"))
        for d in temp["data"]:
            for p in d["paragraphs"]:
                for q in p["qas"]:
                    if q.get("is_impossible", False):
                        dev_labels[q["id"]] = "is_impossible"
                    else:
                        dev_labels[q["id"]] = q["answers"][0]["text"]
        not_included = set(list(dev_labels.keys())) - set(list(predictions_json.keys()))
        if len(not_included) > 0:
            logger.info("There were missing predicitons for question ids: %s", list(not_included))
        for x in not_included:
            predictions_json[x] = ""

    # os.makedirs("model_output", exist_ok=True)
    # filepath = Path("model_output") / out_filename
    json.dump(predictions_json, open(out_filename, "w"))
    logger.info("Written Squad predictions to: %s", out_filename)


def _read_dpr_json(
    file: str,
    max_samples: Optional[int] = None,
    proxies: Optional[Any] = None,
    num_hard_negatives: int = 1,
    num_positives: int = 1,
    shuffle_negatives: bool = True,
    shuffle_positives: bool = False,
):
    """
    Reads a Dense Passage Retrieval (DPR) data file in json format and returns a list of dictionaries.

    :param file: filename of DPR data in json format

    Returns:
        list of dictionaries: List[dict]
        each dictionary: {
                    "query": str -> query_text
                    "passages": List[dictionaries] -> [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
                                {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
                                ...]
                    }
        example:
                ["query": 'who sings does he love me with reba'
                "passages" : [{'title': 'Does He Love You',
                    'text': 'Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba\'s album "Greatest Hits Volume Two". It is one of country music\'s several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members',
                    'label': 'positive',
                    'external_id': '11828866'},
                    {'title': 'When the Nightingale Sings',
                    'text': "When the Nightingale Sings When The Nightingale Sings is a Middle English poem, author unknown, recorded in the British Library's Harley 2253 manuscript, verse 25. It is a love poem, extolling the beauty and lost love of an unknown maiden. When þe nyhtegale singes þe wodes waxen grene.<br> Lef ant gras ant blosme springes in aueryl y wene,<br> Ant love is to myn herte gon wiþ one spere so kene<br> Nyht ant day my blod hit drynkes myn herte deþ me tene. Ich have loved al þis er þat y may love namore,<br> Ich have siked moni syk lemmon for",
                    'label': 'hard_negative',
                    'external_id': '10891637'}]
                ]

    """
    # get remote dataset if needed
    if not os.path.exists(file):
        logger.info("Couldn't find %s locally. Trying to download ...", file)
        _download_extract_downstream_data(file, proxies=proxies)

    if Path(file).suffix.lower() == ".jsonl":
        dicts = []
        with open(file, encoding="utf-8") as f:
            for line in f:
                dicts.append(json.loads(line))
    else:
        dicts = json.load(open(file, encoding="utf-8"))

    if max_samples:
        dicts = random.sample(dicts, min(max_samples, len(dicts)))

    # convert DPR dictionary to standard dictionary
    query_json_keys = ["question", "questions", "query"]
    positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
    hard_negative_json_keys = [
        "hard_negative_contexts",
        "hard_negative_ctxs",
        "hard_negative_context",
        "hard_negative_ctx",
    ]
    standard_dicts = []
    for dict in dicts:
        sample = {}
        passages = []
        for key, val in dict.items():
            if key in query_json_keys:
                sample["query"] = val
            elif key in positive_context_json_keys:
                if shuffle_positives:
                    random.shuffle(val)
                for passage in val[:num_positives]:
                    passages.append(
                        {
                            "title": passage["title"],
                            "text": passage["text"],
                            "label": "positive",
                            "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8]),
                        }
                    )
            elif key in hard_negative_json_keys:
                if shuffle_negatives:
                    random.shuffle(val)
                for passage in val[:num_hard_negatives]:
                    passages.append(
                        {
                            "title": passage["title"],
                            "text": passage["text"],
                            "label": "hard_negative",
                            "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8]),
                        }
                    )
        sample["passages"] = passages
        standard_dicts.append(sample)
    return standard_dicts


def _read_squad_file(filename: str, proxies=None):
    """Read a SQuAD json file"""
    if not os.path.exists(filename):
        logger.info("Couldn't find %s locally. Trying to download ...", filename)
        _download_extract_downstream_data(filename, proxies)
    with open(filename, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    return input_data


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def _download_extract_downstream_data(input_file: str, proxies=None):
    # download archive to temp dir and extract to correct position
    full_path = Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info("downloading and extracting file {} to dir {}".format(taskname, datadir))
    if taskname not in DOWNSTREAM_TASK_MAP:
        logger.error("Cannot download {}. Unknown data source.".format(taskname))
    else:
        if os.name == "nt":  # make use of NamedTemporaryFile compatible with Windows
            delete_tmp_file = False
        else:
            delete_tmp_file = True
        with tempfile.NamedTemporaryFile(delete=delete_tmp_file) as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here


def _is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False
