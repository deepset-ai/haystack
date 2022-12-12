from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Union

import hashlib
import json
import logging
import random
from itertools import groupby
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.processor import Processor
from haystack.utils.experiment_tracking import Tracker as tracker
from haystack.modeling.visual import TRACTOR_SMALL

if TYPE_CHECKING:
    from haystack.nodes import FARMReader

logger = logging.getLogger(__name__)


class DataSilo:
    """Generates and stores PyTorch DataLoader objects for the train, dev and test datasets.
    Relies upon functionality in the processor to do the conversion of the data. Will also
    calculate and display some statistics.
    """

    def __init__(
        self,
        processor: Processor,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
        distributed: bool = False,
        automatic_loading: bool = True,
        max_multiprocessing_chunksize: int = 512,
        max_processes: int = 128,
        multiprocessing_strategy: Optional[str] = None,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
    ):
        """
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :param batch_size: The size of batch that should be returned by the DataLoader for the training set.
        :param eval_batch_size: The size of batch that should be returned by the DataLoaders for the dev and test set.
        :param distributed: Set to True if you are running in a distributed evn, e.g. using DistributedDataParallel.
                            The DataSilo will init the DataLoader with a DistributedSampler() to distribute batches.
        :param automatic_loading: Set to False, if you don't want to automatically load data at initialization.
        :param max_multiprocessing_chunksize: max possible value for chunksize as calculated by `calc_chunksize()`
            in `haystack.basics.utils`. For certain cases like lm_finetuning, a smaller value can be set, as the default chunksize
            values are rather large that might cause memory issues.
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
                              .. deprecated:: 1.9
                                    Multiprocessing has been removed in 1.9. This parameter will be ignored.
        :multiprocessing_strategy: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
                                   If your system has low limits for the number of open file descriptors, and you can’t raise them,
                                   you should use the file_system strategy.
                                   .. deprecated:: 1.9
                                        Multiprocessing has been removed in 1.9. This parameter will be ignored.
        :param caching: save the processed datasets on disk to save time/compute if the same train data is used to run
                        multiple experiments. Each cache has a checksum based on the train_filename of the Processor
                        and the batch size.
        :param cache_path: root dir for storing the datasets' cache.
        """
        self.distributed = distributed
        self.processor = processor
        self.data = {}  # type: Dict
        self.batch_size = batch_size
        self.class_weights = None
        self.max_processes = max_processes
        self.multiprocessing_strategy = multiprocessing_strategy
        self.max_multiprocessing_chunksize = max_multiprocessing_chunksize
        self.caching = caching
        self.cache_path = cache_path
        self.tensor_names = None
        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size

        if len(self.processor.tasks) == 0:
            raise Exception(
                "No task initialized. Try initializing the processor with a metric and a label list. "
                "Alternatively you can add a task using Processor.add_task()"
            )

        loaded_from_cache = False
        if self.caching:  # Check if DataSets are present in cache
            checksum = self._get_checksum()
            dataset_path = self.cache_path / checksum

            if dataset_path.exists():
                self._load_dataset_from_cache(dataset_path)
                loaded_from_cache = True

        if not loaded_from_cache and automatic_loading:
            # In most cases we want to load all data automatically, but in some cases we rather want to do this
            # later or load from dicts instead of file
            self._load_data()

    def _get_dataset(self, filename: Optional[Union[str, Path]], dicts: Optional[List[Dict]] = None):
        if not filename and not dicts:
            raise ValueError("You must either supply `filename` or `dicts`")

        # loading dicts from file (default)
        if dicts is None:
            dicts = list(self.processor.file_to_dicts(filename))  # type: ignore
            # shuffle list of dicts here if we later want to have a random dev set splitted from train set
            if str(self.processor.train_filename) in str(filename):
                if not self.processor.dev_filename:
                    if self.processor.dev_split > 0.0:
                        random.shuffle(dicts)

        num_dicts = len(dicts)
        datasets = []
        problematic_ids_all = set()
        batch_size = self.max_multiprocessing_chunksize
        for i in tqdm(range(0, num_dicts, batch_size), desc="Preprocessing dataset", unit=" Dicts"):
            processing_batch = dicts[i : i + batch_size]
            dataset, tensor_names, problematic_sample_ids = self.processor.dataset_from_dicts(
                dicts=processing_batch, indices=list(range(len(processing_batch)))  # TODO remove indices
            )
            datasets.append(dataset)
            problematic_ids_all.update(problematic_sample_ids)

        self.processor.log_problematic(problematic_ids_all)
        datasets = [d for d in datasets if d]
        concat_datasets = ConcatDataset(datasets)  # type: Dataset
        return concat_datasets, tensor_names

    def _load_data(
        self,
        train_dicts: Optional[List[Dict]] = None,
        dev_dicts: Optional[List[Dict]] = None,
        test_dicts: Optional[List[Dict]] = None,
    ):
        """
        Loading the train, dev and test datasets either from files (default) or from supplied dicts.
        The processor is called to handle the full conversion from "raw data" to a Pytorch Dataset.
        The resulting datasets are loaded into DataSilo.data

        :param train_dicts: (Optional) dicts containing examples for training.
        :param dev_dicts: (Optional) dicts containing examples for dev.
        :param test_dicts: (Optional) dicts containing examples for test.
        :return: None
        """

        logger.info("\nLoading data into the data silo ..." "{}".format(TRACTOR_SMALL))
        # train data
        logger.info("LOADING TRAIN DATA")
        logger.info("==================")
        if train_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["train"], self.tensor_names = self._get_dataset(filename=None, dicts=train_dicts)
        elif self.processor.train_filename:
            # or from a file (default)
            train_file = self.processor.data_dir / self.processor.train_filename
            logger.info("Loading train set from: {} ".format(train_file))
            self.data["train"], self.tensor_names = self._get_dataset(train_file)
        else:
            logger.info("No train set is being loaded")
            self.data["train"] = None

        # dev data
        logger.info("")
        logger.info("LOADING DEV DATA")
        logger.info("=================")
        if dev_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["dev"], self.tensor_names = self._get_dataset(filename=None, dicts=dev_dicts)
        elif self.processor.dev_filename:
            # or from file (default)
            dev_file = self.processor.data_dir / self.processor.dev_filename
            logger.info("Loading dev set from: {}".format(dev_file))
            self.data["dev"], _ = self._get_dataset(dev_file)
        elif self.processor.dev_split > 0.0:
            # or split it apart from train set
            logger.info("Loading dev set as a slice of train set")
            self._create_dev_from_train()
        else:
            logger.info("No dev set is being loaded")
            self.data["dev"] = None

        logger.info("")
        logger.info("LOADING TEST DATA")
        logger.info("=================")
        # test data
        if test_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["test"], self.tensor_names = self._get_dataset(filename=None, dicts=test_dicts)
        elif self.processor.test_filename:
            # or from file (default)
            test_file = self.processor.data_dir / self.processor.test_filename
            logger.info("Loading test set from: {}".format(test_file))
            if self.tensor_names:
                self.data["test"], _ = self._get_dataset(test_file)
            else:
                self.data["test"], self.tensor_names = self._get_dataset(test_file)
        else:
            logger.info("No test set is being loaded")
            self.data["test"] = None

        if self.caching:
            self._save_dataset_to_cache()

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _load_dataset_from_cache(self, cache_dir: Path):
        """
        Load serialized dataset from a cache.
        """
        logger.info("Loading datasets from cache at %s", cache_dir)
        self.data["train"] = torch.load(cache_dir / "train_dataset")

        dev_dataset_path = cache_dir / "dev_dataset"
        if dev_dataset_path.exists():
            self.data["dev"] = torch.load(dev_dataset_path)
        else:
            self.data["dev"] = None

        test_dataset_path = cache_dir / "test_dataset"
        if test_dataset_path.exists():
            self.data["test"] = torch.load(test_dataset_path)
        else:
            self.data["test"] = None

        self.tensor_names = torch.load(cache_dir / "tensor_names")

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _get_checksum(self):
        """
        Get checksum based on a dict to ensure validity of cached DataSilo
        """
        # keys in the dict identifies uniqueness for a given DataSilo.
        payload_dict = {
            "train_filename": str(Path(self.processor.train_filename).absolute()),
            "data_dir": str(self.processor.data_dir.absolute()),
            "max_seq_len": self.processor.max_seq_len,
            "dev_split": self.processor.dev_split,
            "tasks": self.processor.tasks,
        }
        checksum = get_dict_checksum(payload_dict)
        return checksum

    def _save_dataset_to_cache(self):
        """
        Serialize and save dataset to a cache.
        """
        checksum = self._get_checksum()

        cache_dir = self.cache_path / checksum
        cache_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.data["train"], cache_dir / "train_dataset")

        if self.data["dev"]:
            torch.save(self.data["dev"], cache_dir / "dev_dataset")

        if self.data["test"]:
            torch.save(self.data["test"], cache_dir / "test_dataset")

        torch.save(self.tensor_names, cache_dir / "tensor_names")
        logger.info("Cached the datasets at %s", cache_dir)

    def _initialize_data_loaders(self):
        """
        Initializing train, dev and test data loaders for the already loaded datasets.
        """

        if self.data["train"] is not None:
            if self.distributed:
                sampler_train = DistributedSampler(self.data["train"])
            else:
                sampler_train = RandomSampler(self.data["train"])

            data_loader_train = NamedDataLoader(
                dataset=self.data["train"],
                sampler=sampler_train,
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_train = None

        if self.data["dev"] is not None:
            data_loader_dev = NamedDataLoader(
                dataset=self.data["dev"],
                sampler=SequentialSampler(self.data["dev"]),
                batch_size=self.eval_batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_dev = None

        if self.data["test"] is not None:
            data_loader_test = NamedDataLoader(
                dataset=self.data["test"],
                sampler=SequentialSampler(self.data["test"]),
                batch_size=self.eval_batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_test = None

        self.loaders = {"train": data_loader_train, "dev": data_loader_dev, "test": data_loader_test}

    def _create_dev_from_train(self):
        """
        Split a dev set apart from the train dataset.
        """
        n_dev = int(self.processor.dev_split * len(self.data["train"]))
        n_train = len(self.data["train"]) - n_dev

        train_dataset, dev_dataset = self.random_split_ConcatDataset(self.data["train"], lengths=[n_train, n_dev])
        self.data["train"] = train_dataset
        if len(dev_dataset) > 0:
            self.data["dev"] = dev_dataset
        else:
            logger.warning("No dev set created. Please adjust the dev_split parameter.")

        logger.info(
            f"Took {len(dev_dataset)} samples out of train set to create dev set (dev split is roughly {self.processor.dev_split})"
        )

    def random_split_ConcatDataset(self, ds: ConcatDataset, lengths: List[int]):
        """
        Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
        Samples inside Concatdataset should already be shuffled.

        :param ds: Dataset to be split.
        :param lengths: Lengths of splits to be produced.
        """
        if sum(lengths) != len(ds):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        try:
            idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
        except IndexError:
            raise Exception(
                "All dataset chunks are being assigned to train set leaving no samples for dev set. "
                "Either consider increasing dev_split or setting it to 0.0\n"
                f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
                f"train/dev split: {lengths}"
            )

        assert idx_dataset >= 1, (
            "Dev_split ratio is too large, there is no data in train set. "
            f"Please lower dev_split = {self.processor.dev_split}"
        )

        train = ConcatDataset(ds.datasets[:idx_dataset])  # type: Dataset
        test = ConcatDataset(ds.datasets[idx_dataset:])  # type: Dataset
        return train, test

    def _calculate_statistics(self):
        """Calculate and log simple summary statistics of the datasets"""
        logger.info("")
        logger.info("DATASETS SUMMARY")
        logger.info("================")

        self.counts = {}
        clipped = -1
        ave_len = -1

        if self.data["train"]:
            self.counts["train"] = len(self.data["train"])
            if "input_ids" in self.tensor_names:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_single_encoder()
            elif "query_input_ids" in self.tensor_names and "passage_input_ids" in self.tensor_names:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_biencoder()
            else:
                logger.warning(
                    f"Could not compute length statistics because 'input_ids' or 'query_input_ids' and 'passage_input_ids' are missing."
                )
                clipped = -1
                ave_len = -1
        else:
            self.counts["train"] = 0

        if self.data["dev"]:
            self.counts["dev"] = len(self.data["dev"])
        else:
            self.counts["dev"] = 0

        if self.data["test"]:
            self.counts["test"] = len(self.data["test"])
        else:
            self.counts["test"] = 0

        logger.info("Examples in train: {}".format(self.counts["train"]))
        logger.info("Examples in dev  : {}".format(self.counts["dev"]))
        logger.info("Examples in test : {}".format(self.counts["test"]))
        logger.info("Total examples   : {}".format(self.counts["train"] + self.counts["dev"] + self.counts["test"]))
        logger.info("")
        if self.data["train"]:
            if "input_ids" in self.tensor_names:
                logger.info("Longest sequence length observed after clipping:     {}".format(max(seq_lens)))
                logger.info("Average sequence length after clipping: {}".format(ave_len))
                logger.info("Proportion clipped:      {}".format(clipped))
                if clipped > 0.5:
                    logger.info(
                        f"[Haystack Tip] {round(clipped * 100, 1)}% of your samples got cut down to {max_seq_len} tokens. "
                        "Consider increasing max_seq_len "
                        f"(the maximum value allowed with the current model is max_seq_len={self.processor.tokenizer.model_max_length}, "
                        "if this is not enough consider splitting the document in smaller units or changing the model). "
                        "This will lead to higher memory consumption but is likely to improve your model performance"
                    )
            elif "query_input_ids" in self.tensor_names and "passage_input_ids" in self.tensor_names:
                logger.info(
                    "Longest query length observed after clipping: {}   - for max_query_len: {}".format(
                        max(seq_lens[0]), max_seq_len[0]
                    )
                )
                logger.info("Average query length after clipping:          {}".format(ave_len[0]))
                logger.info("Proportion queries clipped:                   {}".format(clipped[0]))
                logger.info("")
                logger.info(
                    "Longest passage length observed after clipping: {}   - for max_passage_len: {}".format(
                        max(seq_lens[1]), max_seq_len[1]
                    )
                )
                logger.info("Average passage length after clipping:          {}".format(ave_len[1]))
                logger.info("Proportion passages clipped:                    {}".format(clipped[1]))

        tracker.track_params(
            {
                "n_samples_train": self.counts["train"],
                "n_samples_dev": self.counts["dev"],
                "n_samples_test": self.counts["test"],
                "batch_size": self.batch_size,
                "ave_seq_len": ave_len,
                "clipped": clipped,
            }
        )

    def _calc_length_stats_single_encoder(self):
        seq_lens = []
        for dataset in self.data["train"].datasets:
            train_input_numpy = dataset[:][self.tensor_names.index("input_ids")].numpy()
            seq_lens.extend(np.sum(train_input_numpy != self.processor.tokenizer.pad_token_id, axis=1))
        max_seq_len = dataset[:][self.tensor_names.index("input_ids")].shape[1]
        clipped = np.mean(np.array(seq_lens) == max_seq_len) if seq_lens else 0
        ave_len = np.mean(seq_lens) if seq_lens else 0
        return clipped, ave_len, seq_lens, max_seq_len

    def _calc_length_stats_biencoder(self):
        seq_lens = [[], []]
        for dataset in self.data["train"].datasets:
            query_input_numpy = dataset[:][self.tensor_names.index("query_input_ids")].numpy()
            num_passages = dataset[:][self.tensor_names.index("passage_input_ids")].shape[1]
            bs = dataset[:][self.tensor_names.index("passage_input_ids")].shape[0]
            passage_input_numpy = (
                dataset[:][self.tensor_names.index("passage_input_ids")].numpy().reshape((bs, -1), order="C")
            )
            qlen = np.sum(query_input_numpy != self.processor.query_tokenizer.pad_token_id, axis=1)
            plen = np.sum(passage_input_numpy != self.processor.passage_tokenizer.pad_token_id, axis=1) / num_passages
            seq_lens[0].extend(qlen)
            seq_lens[1].extend(plen)
        q_max_seq_len = dataset[:][self.tensor_names.index("query_input_ids")].shape[1]
        p_max_seq_len = dataset[:][self.tensor_names.index("passage_input_ids")].shape[2]
        clipped_q = np.mean(np.array(seq_lens[0]) == q_max_seq_len) if seq_lens[0] else 0
        ave_len_q = np.mean(seq_lens[0]) if seq_lens[0] else 0
        clipped_p = np.mean(np.array(seq_lens[1]) == p_max_seq_len) if seq_lens[1] else 0
        ave_len_p = np.mean(seq_lens[1]) if seq_lens[1] else 0
        clipped = [clipped_q, clipped_p]
        ave_len = [ave_len_q, ave_len_p]
        max_seq_len = [q_max_seq_len, p_max_seq_len]
        return clipped, ave_len, seq_lens, max_seq_len

    def get_data_loader(self, dataset_name: str):
        """
        Returns data loader for specified split of dataset.

        :param dataset_name: Split of dataset. Either 'train' or 'dev' or 'test'.
        """
        return self.loaders[dataset_name]

    def n_samples(self, dataset_name: str):
        """
        Returns the number of samples in a given dataset.

        :param dataset_name: Split of dataset. Choose from 'train', 'dev' or 'test'.
        """
        return self.counts[dataset_name]


class DataSiloForCrossVal:
    """
    Perform cross validation or nested cross validation.

    For performing cross validation or nested cross validation, we really want to combine all the
    instances from all the sets or just some of the sets, then create a different data silo
    instance for each fold or nested fold.
    Calling DataSiloForCrossVal.make() creates a list of DataSiloForCrossVal instances - one for each fold.
    """

    def __init__(
        self,
        origsilo: DataSilo,
        trainset: Union[List, Dataset],
        devset: Union[List, Dataset],
        testset: Union[List, Dataset],
    ):
        self.tensor_names = origsilo.tensor_names
        self.data = {"train": trainset, "dev": devset, "test": testset}
        self.processor = origsilo.processor
        self.batch_size = origsilo.batch_size
        # should not be necessary, xval makes no sense with huge data
        # sampler_train = DistributedSampler(self.data["train"])
        sampler_train = RandomSampler(trainset)

        self.data_loader_train = NamedDataLoader(
            dataset=trainset, sampler=sampler_train, batch_size=self.batch_size, tensor_names=self.tensor_names
        )
        self.data_loader_dev = NamedDataLoader(
            dataset=devset,
            sampler=SequentialSampler(devset),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.data_loader_test = NamedDataLoader(
            dataset=testset,
            sampler=SequentialSampler(testset),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.loaders = {"train": self.data_loader_train, "dev": self.data_loader_dev, "test": self.data_loader_test}

    def get_data_loader(self, which):
        return self.loaders[which]

    @classmethod
    def make(
        cls,
        datasilo: DataSilo,
        sets: List[str] = ["train", "dev", "test"],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratified: bool = True,
        n_neg_answers_per_question: int = 1,
        n_inner_splits: Optional[int] = None,
    ):
        """
        Create number of folds data-silo-like objects which can be used for training from the
        original data silo passed on.

        :param datasilo: The data silo that contains the original data.
        :param sets: Which sets to use to create the xval folds (strings)
        :param n_splits: number of folds to create
        :param shuffle: shuffle each class' samples before splitting
        :param random_state: random state for shuffling
        :param stratified: If class stratification should be done.
            It is never done with question answering.
        :param n_neg_answers_per_question: number of negative answers per question to include for training
        """
        if "question_answering" in datasilo.processor.tasks and n_inner_splits is None:  # type: ignore
            return cls._make_question_answering(
                datasilo, sets, n_splits, shuffle, random_state, n_neg_answers_per_question
            )

        else:
            raise RuntimeError("Cross validation can not be done under these conditions!")

    @classmethod
    def _make_question_answering(
        cls,
        datasilo: DataSilo,
        sets: List[str] = ["train", "dev", "test"],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        n_neg_answers_per_question: int = 1,
    ):
        """
        Create number of folds data-silo-like objects which can be used for training from the
        original data silo passed on. This function takes into account the characteristics of the
        data for question-answering-

        :param datasilo: The data silo that contains the original data.
        :param sets: Which sets to use to create the xval folds (strings).
        :param n_splits: Number of folds to create.
        :param shuffle: Shuffle each class' samples before splitting.
        :param random_state: Random state for shuffling.
        :param n_neg_answers_per_question: Number of negative answers per question to include for training.
        """
        assert "id" in datasilo.tensor_names, f"Expected tensor 'id' in tensor names, found {datasilo.tensor_names}"  # type: ignore
        assert "labels" in datasilo.tensor_names, f"Expected tensor 'labels' in tensor names, found {datasilo.tensor_names}"  # type: ignore

        id_index = datasilo.tensor_names.index("id")  # type: ignore
        label_index = datasilo.tensor_names.index("labels")  # type:ignore

        sets_to_concat = []
        for setname in sets:
            if datasilo.data[setname]:
                sets_to_concat.extend(datasilo.data[setname])
        all_data = ConcatDataset(sets_to_concat)  # type: Dataset

        documents = []
        keyfunc = lambda x: x[id_index][0]  # pylint: disable=unnecessary-lambda-assignment
        all_data = sorted(all_data.datasets, key=keyfunc)  # type: ignore
        for key, document in groupby(all_data, key=keyfunc):  # type: ignore
            documents.append(list(document))

        xval_split = cls._split_for_qa(
            documents=documents, id_index=id_index, n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        silos = []

        for train_set, test_set in xval_split:
            # Each training set is further divided into actual train and dev set
            if datasilo.processor.dev_split > 0:
                dev_split = datasilo.processor.dev_split
                n_dev = int(np.ceil(dev_split * len(train_set)))
                assert n_dev > 0, f"dev split of {dev_split} is not large enough to split away a development set"
                n_actual_train = len(train_set) - n_dev
                actual_train_set = train_set[:n_actual_train]
                dev_set = train_set[n_actual_train:]
                ds_dev = [sample for document in dev_set for sample in document]
            else:
                ds_dev = None  # type: ignore
                actual_train_set = train_set

            train_samples = []
            for doc in actual_train_set:
                keyfunc = lambda x: x[id_index][1]  # pylint: disable=unnecessary-lambda-assignment
                doc = sorted(doc, key=keyfunc)
                for key, question in groupby(doc, key=keyfunc):
                    # add all available answrs to train set
                    sample_list = list(question)
                    neg_answer_idx: List[int] = []
                    for index, sample in enumerate(sample_list):
                        if sample[label_index][0][0] or sample[label_index][0][1]:
                            train_samples.append(sample)
                        else:
                            neg_answer_idx.append(index)
                    # add random n_neg_answers_per_question samples to train set
                    if len(neg_answer_idx) <= n_neg_answers_per_question:
                        train_samples.extend([sample_list[idx] for idx in neg_answer_idx])
                    else:
                        neg_answer_idx = random.sample(neg_answer_idx, n_neg_answers_per_question)
                        train_samples.extend(
                            # For some reason pylint seems to be just wrong here. It's therefore silenced.
                            # Check if the issue persists in case of a future refactoring.
                            [sample_list[idx] for idx in neg_answer_idx]  # pylint: disable=invalid-sequence-index
                        )

            ds_train = train_samples
            ds_test = [sample for document in test_set for sample in document]
            silos.append(DataSiloForCrossVal(datasilo, ds_train, ds_dev, ds_test))
        return silos

    @staticmethod
    def _split_for_qa(
        documents: List, id_index: int, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None
    ):
        keyfunc = lambda x: x[id_index][1]  # pylint: disable=unnecessary-lambda-assignment
        if shuffle:
            fixed_random = random.Random()
            fixed_random.seed(random_state)
            fixed_random.shuffle(documents)

        questions_per_doc = []
        for doc in documents:
            # group samples in current doc by question id
            doc = sorted(doc, key=keyfunc)
            questions = list(groupby(doc, key=keyfunc))
            questions_per_doc.append(len(questions))

        # split documents into n_splits splits with approximately same number of questions per split
        questions_per_doc = np.array(questions_per_doc)
        accumulated_questions_per_doc = questions_per_doc.cumsum()  # type: ignore
        questions_per_fold = accumulated_questions_per_doc[-1] // n_splits
        accumulated_questions_per_fold = np.array(range(1, n_splits)) * questions_per_fold
        if accumulated_questions_per_fold[0] < accumulated_questions_per_doc[0]:
            accumulated_questions_per_fold[0] = accumulated_questions_per_doc[0] + 1
        indices_to_split_at = np.searchsorted(
            accumulated_questions_per_doc, accumulated_questions_per_fold, side="right"
        )
        splits = np.split(documents, indices_to_split_at)

        for split in splits:
            assert len(split) > 0

        for idx, split in enumerate(splits):
            current_test_set = split
            current_train_set = np.hstack(np.delete(splits, idx, axis=0))

            yield current_train_set, current_test_set


def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(json.dumps(payload_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return checksum


class DistillationDataSilo(DataSilo):
    """
    This data silo does a forward pass on the full data set on a teacher model for model distillation.
    As its done in preprocessing, it does not need to be repeated in each epoch and can be cached.
    """

    def __init__(
        self,
        teacher_model: "FARMReader",
        teacher_batch_size: int,
        device: torch.device,
        processor: Processor,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
        distributed: bool = False,
        automatic_loading: bool = True,
        max_processes: int = 128,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
    ):
        self.teacher = teacher_model
        self.teacher_batch_size = teacher_batch_size
        self.device = device
        max_processes = 1  # fix as long as multithreading is not working with teacher attribute
        super().__init__(
            max_processes=max_processes,
            processor=processor,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            distributed=distributed,
            automatic_loading=automatic_loading,
            caching=caching,
            cache_path=cache_path,
        )

    def _run_teacher(self, batch: dict) -> List[torch.Tensor]:
        """
        Run the teacher model on the given batch.
        """
        params = {
            "input_ids": batch["input_ids"],
            "segment_ids": batch["segment_ids"],
            "padding_mask": batch["padding_mask"],
        }
        if "output_hidden_states" in batch.keys():
            params["output_hidden_states"] = batch["output_hidden_states"]
        if "output_attentions" in batch.keys():
            params["output_attentions"] = batch["output_attentions"]
        return self.teacher.inferencer.model(**params)

    def _pass_batches(
        self,
        batch: List[List[torch.Tensor]],
        corresponding_chunks: List[int],
        teacher_outputs: List[List[Tuple[torch.Tensor, ...]]],
        tensor_names: List[str],
    ):
        with torch.inference_mode():
            batch_transposed = zip(*batch)  # transpose dimensions (from batch, features, ... to features, batch, ...)
            batch_transposed_list = [torch.stack(b) for b in batch_transposed]  # create tensors for each feature
            batch_dict = {
                key: tensor.to(self.device) for key, tensor in zip(tensor_names, batch_transposed_list)
            }  # create input dict
            y = self._run_teacher(batch=batch_dict)  # run teacher model
            y = [y.cpu() for y in y]
            self.output_len = len(y)

            # grouping by chunk
            for i, data in zip(corresponding_chunks, zip(*y)):  # transpose back
                teacher_outputs[i].append(data)
            return

    def _teacher_output_names(self) -> List[str]:
        return ["teacher_output_" + str(i) for i in range(self.output_len)]

    def _get_dataset(self, filename: Optional[Union[str, Path]], dicts: Optional[List[Dict]] = None):
        concat_datasets, tensor_names = super()._get_dataset(filename, dicts)

        batch = []
        corresponding_chunks = (
            []
        )  # to be able to associate elements of batches with chunks (elements could be from multiple chunks)

        teacher_outputs: List[List[Tuple[torch.Tensor, ...]]] = []  # list of teacher outputs group in list by chunk

        # creating batches from chunks
        for i, dataset in enumerate(tqdm(concat_datasets.datasets, desc="Doing forward pass on teacher model")):
            teacher_outputs.append([])
            for x in zip(*dataset.tensors):  # loop through chunks
                batch.append(x)
                corresponding_chunks.append(i)
                if len(batch) == self.teacher_batch_size:
                    self._pass_batches(
                        batch, corresponding_chunks, teacher_outputs, tensor_names
                    )  # doing forward pass on teacher model
                    batch = []
                    corresponding_chunks = []
        if batch:
            self._pass_batches(batch, corresponding_chunks, teacher_outputs, tensor_names)

        # appending teacher outputs to original dataset
        for dataset, teacher_output in zip(concat_datasets.datasets, teacher_outputs):
            dataset.tensors += tuple(torch.stack(tensors) for tensors in zip(*teacher_output))
        tensor_names += self._teacher_output_names()
        concat_datasets = ConcatDataset(concat_datasets.datasets)  # making sure metrics are updated
        return concat_datasets, tensor_names

    def _get_checksum(self):
        """
        Get checksum based on a dict to ensure validity of cached DataSilo
        """
        # keys in the dict identifies uniqueness for a given DataSilo.
        payload_dict = {
            "train_filename": str(Path(self.processor.train_filename).absolute()),
            "data_dir": str(self.processor.data_dir.absolute()),
            "max_seq_len": self.processor.max_seq_len,
            "dev_split": self.processor.dev_split,
            "tasks": self.processor.tasks,
            "teacher_name_or_path": self.teacher.model_name_or_path,
            "data_silo_type": self.__class__.__name__,
        }
        checksum = get_dict_checksum(payload_dict)
        return checksum
