from typing import List, Dict, Union, Optional, Any

import logging
from pathlib import Path
from copy import deepcopy
from requests.exceptions import HTTPError

import numpy as np
from tqdm.auto import tqdm

import torch
from torch.nn import DataParallel
from torch.utils.data.sampler import SequentialSampler
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever._embedding_encoder import _EMBEDDING_ENCODERS
from haystack.modeling.model.tokenization import Tokenizer
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.modeling.model.triadaptive_model import TriAdaptiveModel
from haystack.modeling.model.prediction_head import TextSimilarityHead
from haystack.modeling.data_handler.processor import TextSimilarityProcessor, TableTextSimilarityProcessor
from haystack.modeling.data_handler.data_silo import DataSilo
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.model.optimization import initialize_optimizer
from haystack.modeling.training.base import Trainer
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


class DensePassageRetriever(BaseRetriever):
    """
    Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
    See the original paper for more details:
    Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
    (https://arxiv.org/abs/2004.04906).
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base",
        model_version: Optional[str] = None,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        infer_tokenizer_classes: bool = False,
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format

        **Example:**

                ```python
                |    # remote model from FAIR
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                |                          passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
                |    # or from local path
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="model_directory/question-encoder",
                |                          passage_embedding_model="model_directory/context-encoder")
                ```

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models
                                      Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by hugging-face transformers' modelhub models
                                        Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        :param use_fast_tokenizers: Whether to use fast Rust tokenizers
        :param infer_tokenizer_classes: Whether to infer tokenizer class from the model config / name.
                                        If `False`, the class always loads `DPRQuestionEncoderTokenizer` and `DPRContextEncoderTokenizer`.
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: as multi-GPU training is currently not implemented for DPR, training
                        will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        super().__init__()

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.scale_score = scale_score

        if document_store is None:
            logger.warning(
                "DensePassageRetriever initialized without a document store. "
                "This is fine if you are performing DPR training. "
                "Otherwise, please provide a document store in the constructor."
            )
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore"
            )

        self.infer_tokenizer_classes = infer_tokenizer_classes
        tokenizers_default_classes = {"query": "DPRQuestionEncoderTokenizer", "passage": "DPRContextEncoderTokenizer"}
        if self.infer_tokenizer_classes:
            tokenizers_default_classes["query"] = None  # type: ignore
            tokenizers_default_classes["passage"] = None  # type: ignore

        # Init & Load Encoders
        self.query_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=query_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["query"],
            use_auth_token=use_auth_token,
        )
        self.query_encoder = LanguageModel.load(
            pretrained_model_name_or_path=query_embedding_model,
            revision=model_version,
            language_model_class="DPRQuestionEncoder",
            use_auth_token=use_auth_token,
        )
        self.passage_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=passage_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["passage"],
            use_auth_token=use_auth_token,
        )
        self.passage_encoder = LanguageModel.load(
            pretrained_model_name_or_path=passage_embedding_model,
            revision=model_version,
            language_model_class="DPRContextEncoder",
            use_auth_token=use_auth_token,
        )

        self.processor = TextSimilarityProcessor(
            query_tokenizer=self.query_tokenizer,
            passage_tokenizer=self.passage_tokenizer,
            max_seq_len_passage=max_seq_len_passage,
            max_seq_len_query=max_seq_len_query,
            label_list=["hard_negative", "positive"],
            metric="text_similarity_metric",
            embed_title=embed_title,
            num_hard_negatives=0,
            num_positives=1,
        )
        prediction_head = TextSimilarityHead(
            similarity_function=similarity_function, global_loss_buffer_size=global_loss_buffer_size
        )
        self.model = BiAdaptiveModel(
            language_model1=self.query_encoder,
            language_model2=self.passage_encoder,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm1_output_types=["per_sequence"],
            lm2_output_types=["per_sequence"],
            device=self.devices[0],
        )

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve_batch() since DensePassageRetriever initialized with document_store=None"
            )
            return [[] * len(queries)]  # type: ignore

        documents = []
        query_embs = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(texts=batch))
        for query_emb, cur_filters in zip(query_embs, filters):
            cur_docs = self.document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index,
                headers=headers,
                scale_score=scale_score,
            )
            documents.append(cur_docs)

        return documents

    def _get_predictions(self, dicts):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dicts: list of dictionaries
        examples:[{'query': "where is florida?"}, {'query': "who wrote lord of the rings?"}, ...]
                [{'passages': [{
                    "title": 'Big Little Lies (TV series)',
                    "text": 'series garnered several accolades. It received..',
                    "label": 'positive',
                    "external_id": '18768923'},
                    {"title": 'Framlingham Castle',
                    "text": 'Castle on the Hill "Castle on the Hill" is a song by English..',
                    "label": 'positive',
                    "external_id": '19930582'}, ...]
        :return: dictionary of embeddings for "passages" and "query"
        """
        dataset, tensor_names, _, baskets = self.processor.dataset_from_dicts(
            dicts, indices=[i for i in range(len(dicts))], return_baskets=True
        )

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        all_embeddings = {"query": [], "passages": []}
        self.model.eval()

        # When running evaluations etc., we don't want a progress bar for every single query
        if len(dataset) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar

        with tqdm(
            total=len(data_loader) * self.batch_size,
            unit=" Docs",
            desc=f"Create embeddings",
            position=1,
            leave=False,
            disable=disable_tqdm,
        ) as progress_bar:
            for batch in data_loader:
                batch = {key: batch[key].to(self.devices[0]) for key in batch}

                # get logits
                with torch.no_grad():
                    query_embeddings, passage_embeddings = self.model.forward(**batch)[0]
                    if query_embeddings is not None:
                        all_embeddings["query"].append(query_embeddings.cpu().numpy())
                    if passage_embeddings is not None:
                        all_embeddings["passages"].append(passage_embeddings.cpu().numpy())
                progress_bar.update(self.batch_size)

        if all_embeddings["passages"]:
            all_embeddings["passages"] = np.concatenate(all_embeddings["passages"])
        if all_embeddings["query"]:
            all_embeddings["query"] = np.concatenate(all_embeddings["query"])
        return all_embeddings

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [{"query": q} for q in texts]
        result = self._get_predictions(queries)["query"]
        return result

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within Haystack.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        if self.processor.num_hard_negatives != 0:
            logger.warning(
                f"'num_hard_negatives' is set to {self.processor.num_hard_negatives}, but inference does "
                f"not require any hard negatives. Setting num_hard_negatives to 0."
            )
            self.processor.num_hard_negatives = 0

        passages = [
            {
                "passages": [
                    {
                        "title": d.meta["name"] if d.meta and "name" in d.meta else "",
                        "text": d.content,
                        "label": d.meta["label"] if d.meta and "label" in d.meta else "positive",
                        "external_id": d.id,
                    }
                ]
            }
            for d in docs
        ]
        embeddings = self._get_predictions(passages)["passages"]

        return embeddings

    def train(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: str = None,
        test_filename: str = None,
        max_samples: int = None,
        max_processes: int = 128,
        multiprocessing_strategy: Optional[str] = None,
        dev_split: float = 0,
        batch_size: int = 2,
        embed_title: bool = True,
        num_hard_negatives: int = 1,
        num_positives: int = 1,
        n_epochs: int = 3,
        evaluate_every: int = 1000,
        n_gpu: int = 1,
        learning_rate: float = 1e-5,
        epsilon: float = 1e-08,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 100,
        grad_acc_steps: int = 1,
        use_amp: str = None,
        optimizer_name: str = "AdamW",
        optimizer_correct_bias: bool = True,
        save_dir: str = "../saved_models/dpr",
        query_encoder_save_dir: str = "query_encoder",
        passage_encoder_save_dir: str = "passage_encoder",
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
    ):
        """
        train a DensePassageRetrieval model
        :param data_dir: Directory where training file, dev file and test file are present
        :param train_filename: training filename
        :param dev_filename: development set filename, file to be used by model in eval step of training
        :param test_filename: test set filename, file to be used by model in test step after training
        :param max_samples: maximum number of input samples to convert. Can be used for debugging a smaller dataset.
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
        :param multiprocessing_strategy: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
                                         If your system has low limits for the number of open file descriptors, and you can’t raise them,
                                         you should use the file_system strategy.
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param batch_size: total number of samples in 1 batch of data
        :param embed_title: whether to concatenate passage title with each passage. The default setting in official DPR embeds passage title with the corresponding passage
        :param num_hard_negatives: number of hard negative passages(passages which are very similar(high score by BM25) to query but do not contain the answer
        :param num_positives: number of positive passages
        :param n_epochs: number of epochs to train the model on
        :param evaluate_every: number of training steps after evaluation is run
        :param n_gpu: number of gpus to train on
        :param learning_rate: learning rate of optimizer
        :param epsilon: epsilon parameter of optimizer
        :param weight_decay: weight decay parameter of optimizer
        :param grad_acc_steps: number of steps to accumulate gradient over before back-propagation is done
        :param use_amp: Whether to use automatic mixed precision (AMP) or not. The options are:
                    "O0" (FP32)
                    "O1" (Mixed Precision)
                    "O2" (Almost FP16)
                    "O3" (Pure FP16).
                    For more information, refer to: https://nvidia.github.io/apex/amp.html
        :param optimizer_name: what optimizer to use (default: AdamW)
        :param num_warmup_steps: number of warmup steps
        :param optimizer_correct_bias: Whether to correct bias in optimizer
        :param save_dir: directory where models are saved
        :param query_encoder_save_dir: directory inside save_dir where query_encoder model files are saved
        :param passage_encoder_save_dir: directory inside save_dir where passage_encoder model files are saved

        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.
        """
        self.processor.embed_title = embed_title
        self.processor.data_dir = Path(data_dir)
        self.processor.train_filename = train_filename
        self.processor.dev_filename = dev_filename
        self.processor.test_filename = test_filename
        self.processor.max_samples = max_samples
        self.processor.dev_split = dev_split
        self.processor.num_hard_negatives = num_hard_negatives
        self.processor.num_positives = num_positives

        if isinstance(self.model, DataParallel):
            self.model.module.connect_heads_with_processor(self.processor.tasks, require_labels=True)
        else:
            self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

        data_silo = DataSilo(
            processor=self.processor,
            batch_size=batch_size,
            distributed=False,
            max_processes=max_processes,
            multiprocessing_strategy=multiprocessing_strategy,
        )

        # 5. Create an optimizer
        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={
                "name": optimizer_name,
                "correct_bias": optimizer_correct_bias,
                "weight_decay": weight_decay,
                "eps": epsilon,
            },
            schedule_opts={"name": "LinearWarmup", "num_warmup_steps": num_warmup_steps},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            grad_acc_steps=grad_acc_steps,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
        )

        # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer.create_or_load_checkpoint(
            model=self.model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
            checkpoint_root_dir=Path(checkpoint_root_dir),
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
        )

        # 7. Let it grow! Watch the tracked metrics live on experiment tracker (e.g. Mlflow)
        trainer.train()

        self.model.save(Path(save_dir), lm1_name=query_encoder_save_dir, lm2_name=passage_encoder_save_dir)
        self.query_tokenizer.save_pretrained(f"{save_dir}/{query_encoder_save_dir}")
        self.passage_tokenizer.save_pretrained(f"{save_dir}/{passage_encoder_save_dir}")

        if len(self.devices) > 1 and not isinstance(self.model, DataParallel):
            self.model = DataParallel(self.model, device_ids=self.devices)

    def save(
        self,
        save_dir: Union[Path, str],
        query_encoder_dir: str = "query_encoder",
        passage_encoder_dir: str = "passage_encoder",
    ):
        """
        Save DensePassageRetriever to the specified directory.

        :param save_dir: Directory to save to.
        :param query_encoder_dir: Directory in save_dir that contains query encoder model.
        :param passage_encoder_dir: Directory in save_dir that contains passage encoder model.
        :return: None
        """
        save_dir = Path(save_dir)
        self.model.save(save_dir, lm1_name=query_encoder_dir, lm2_name=passage_encoder_dir)
        save_dir = str(save_dir)
        self.query_tokenizer.save_pretrained(save_dir + f"/{query_encoder_dir}")
        self.passage_tokenizer.save_pretrained(save_dir + f"/{passage_encoder_dir}")

    @classmethod
    def load(
        cls,
        load_dir: Union[Path, str],
        document_store: BaseDocumentStore,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        query_encoder_dir: str = "query_encoder",
        passage_encoder_dir: str = "passage_encoder",
        infer_tokenizer_classes: bool = False,
    ):
        """
        Load DensePassageRetriever from the specified directory.
        """
        load_dir = Path(load_dir)
        dpr = cls(
            document_store=document_store,
            query_embedding_model=Path(load_dir) / query_encoder_dir,
            passage_embedding_model=Path(load_dir) / passage_encoder_dir,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers,
            similarity_function=similarity_function,
            infer_tokenizer_classes=infer_tokenizer_classes,
        )
        logger.info(f"DPR model loaded from {load_dir}")

        return dpr


class TableTextRetriever(BaseRetriever):
    """
    Retriever that uses a tri-encoder to jointly retrieve among a database consisting of text passages and tables
    (one transformer for query, one transformer for text passages, one transformer for tables).
    See the original paper for more details:
    Kostić, Bogdan, et al. (2021): "Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models"
    (https://arxiv.org/abs/2108.04049),
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "deepset/bert-small-mm_retrieval-question_encoder",
        passage_embedding_model: Union[Path, str] = "deepset/bert-small-mm_retrieval-passage_encoder",
        table_embedding_model: Union[Path, str] = "deepset/bert-small-mm_retrieval-table_encoder",
        model_version: Optional[str] = None,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        max_seq_len_table: int = 256,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name", "section_title", "caption"],
        use_fast_tokenizers: bool = True,
        infer_tokenizer_classes: bool = False,
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models.
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by hugging-face transformers' modelhub models.
        :param table_embedding_model: Local path or remote name of table encoder checkpoint. The format equala the
                                      one used by hugging-face transformers' modelhub models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / table to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param use_fast_tokenizers: Whether to use fast Rust tokenizers
        :param infer_tokenizer_classes: Whether to infer tokenizer class from the model config / name.
                                        If `False`, the class always loads `DPRQuestionEncoderTokenizer` and `DPRContextEncoderTokenizer`.
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: as multi-GPU training is currently not implemented for TableTextRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        super().__init__()

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.embed_meta_fields = embed_meta_fields
        self.scale_score = scale_score

        if document_store is None:
            logger.warning(
                "DensePassageRetriever initialized without a document store. "
                "This is fine if you are performing DPR training. "
                "Otherwise, please provide a document store in the constructor."
            )
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore"
            )

        self.infer_tokenizer_classes = infer_tokenizer_classes
        tokenizers_default_classes = {
            "query": "DPRQuestionEncoderTokenizer",
            "passage": "DPRContextEncoderTokenizer",
            "table": "DPRContextEncoderTokenizer",
        }
        if self.infer_tokenizer_classes:
            tokenizers_default_classes["query"] = None  # type: ignore
            tokenizers_default_classes["passage"] = None  # type: ignore
            tokenizers_default_classes["table"] = None  # type: ignore

        # Init & Load Encoders
        self.query_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=query_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["query"],
            use_auth_token=use_auth_token,
        )
        self.query_encoder = LanguageModel.load(
            pretrained_model_name_or_path=query_embedding_model,
            revision=model_version,
            language_model_class="DPRQuestionEncoder",
            use_auth_token=use_auth_token,
        )
        self.passage_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=passage_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["passage"],
            use_auth_token=use_auth_token,
        )
        self.passage_encoder = LanguageModel.load(
            pretrained_model_name_or_path=passage_embedding_model,
            revision=model_version,
            language_model_class="DPRContextEncoder",
            use_auth_token=use_auth_token,
        )
        self.table_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=table_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["table"],
            use_auth_token=use_auth_token,
        )
        self.table_encoder = LanguageModel.load(
            pretrained_model_name_or_path=table_embedding_model,
            revision=model_version,
            language_model_class="DPRContextEncoder",
            use_auth_token=use_auth_token,
        )

        self.processor = TableTextSimilarityProcessor(
            query_tokenizer=self.query_tokenizer,
            passage_tokenizer=self.passage_tokenizer,
            table_tokenizer=self.table_tokenizer,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            max_seq_len_table=max_seq_len_table,
            label_list=["hard_negative", "positive"],
            metric="text_similarity_metric",
            embed_meta_fields=embed_meta_fields,
            num_hard_negatives=0,
            num_positives=1,
        )

        prediction_head = TextSimilarityHead(
            similarity_function=similarity_function, global_loss_buffer_size=global_loss_buffer_size
        )

        self.model = TriAdaptiveModel(
            language_model1=self.query_encoder,
            language_model2=self.passage_encoder,
            language_model3=self.table_encoder,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm1_output_types=["per_sequence"],
            lm2_output_types=["per_sequence"],
            lm3_output_types=["per_sequence"],
            device=self.devices[0],
        )

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ) -> List[Document]:
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since TableTextRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve_batch() since TableTextRetriever initialized with document_store=None"
            )
            return [[] * len(queries)]  # type: ignore

        documents = []
        query_embs = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(texts=batch))
        for query_emb, cur_filters in zip(query_embs, filters):
            cur_docs = self.document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index,
                headers=headers,
                scale_score=scale_score,
            )
            documents.append(cur_docs)

        return documents

    def _get_predictions(self, dicts: List[Dict]) -> Dict[str, List[np.ndarray]]:
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dicts: list of dictionaries
        examples:[{'query': "where is florida?"}, {'query': "who wrote lord of the rings?"}, ...]
                [{'passages': [{
                    "title": 'Big Little Lies (TV series)',
                    "text": 'series garnered several accolades. It received..',
                    "label": 'positive',
                    "external_id": '18768923'},
                    {"title": 'Framlingham Castle',
                    "text": 'Castle on the Hill "Castle on the Hill" is a song by English..',
                    "label": 'positive',
                    "external_id": '19930582'}, ...]
        :return: dictionary of embeddings for "passages" and "query"
        """

        dataset, tensor_names, _, baskets = self.processor.dataset_from_dicts(
            dicts, indices=[i for i in range(len(dicts))], return_baskets=True
        )

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        all_embeddings: Dict = {"query": [], "passages": []}
        self.model.eval()

        # When running evaluations etc., we don't want a progress bar for every single query
        if dataset and len(dataset) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar

        with tqdm(
            total=len(data_loader) * self.batch_size,
            unit=" Docs",
            desc=f"Create embeddings",
            position=1,
            leave=False,
            disable=disable_tqdm,
        ) as progress_bar:
            for batch in data_loader:
                batch = {key: batch[key].to(self.devices[0]) for key in batch}

                # get logits
                with torch.no_grad():
                    query_embeddings, passage_embeddings = self.model.forward(**batch)[0]
                    if query_embeddings is not None:
                        all_embeddings["query"].append(query_embeddings.cpu().numpy())
                    if passage_embeddings is not None:
                        all_embeddings["passages"].append(passage_embeddings.cpu().numpy())
                progress_bar.update(self.batch_size)

        if all_embeddings["passages"]:
            all_embeddings["passages"] = np.concatenate(all_embeddings["passages"])
        if all_embeddings["query"]:
            all_embeddings["query"] = np.concatenate(all_embeddings["query"])
        return all_embeddings

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [{"query": q} for q in texts]
        result = self._get_predictions(queries)["query"]
        return result

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of text documents and / or tables using the text passage encoder and
        the table encoder.

        :param docs: List of Document objects used to represent documents / passages in
                     a standardized way within Haystack.
        :return: Embeddings of documents / passages. Shape: (batch_size, embedding_dim)
        """

        if self.processor.num_hard_negatives != 0:
            logger.warning(
                f"'num_hard_negatives' is set to {self.processor.num_hard_negatives}, but inference does "
                f"not require any hard negatives. Setting num_hard_negatives to 0."
            )
            self.processor.num_hard_negatives = 0

        model_input = []
        for doc in docs:
            if doc.content_type == "table":
                model_input.append(
                    {
                        "passages": [
                            {
                                "meta": [
                                    doc.meta[meta_field]
                                    for meta_field in self.embed_meta_fields
                                    if meta_field in doc.meta and isinstance(doc.meta[meta_field], str)
                                ],
                                "columns": doc.content.columns.tolist(),  # type: ignore
                                "rows": doc.content.values.tolist(),  # type: ignore
                                "label": doc.meta["label"] if doc.meta and "label" in doc.meta else "positive",
                                "type": "table",
                                "external_id": doc.id,
                            }
                        ]
                    }
                )
            else:
                model_input.append(
                    {
                        "passages": [
                            {
                                "meta": [
                                    doc.meta[meta_field]
                                    for meta_field in self.embed_meta_fields
                                    if meta_field in doc.meta and isinstance(doc.meta[meta_field], str)
                                ],
                                "text": doc.content,
                                "label": doc.meta["label"] if doc.meta and "label" in doc.meta else "positive",
                                "type": "text",
                                "external_id": doc.id,
                            }
                        ]
                    }
                )

        embeddings = self._get_predictions(model_input)["passages"]

        return embeddings

    def train(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: str = None,
        test_filename: str = None,
        max_samples: int = None,
        max_processes: int = 128,
        dev_split: float = 0,
        batch_size: int = 2,
        embed_meta_fields: List[str] = ["page_title", "section_title", "caption"],
        num_hard_negatives: int = 1,
        num_positives: int = 1,
        n_epochs: int = 3,
        evaluate_every: int = 1000,
        n_gpu: int = 1,
        learning_rate: float = 1e-5,
        epsilon: float = 1e-08,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 100,
        grad_acc_steps: int = 1,
        use_amp: str = None,
        optimizer_name: str = "AdamW",
        optimizer_correct_bias: bool = True,
        save_dir: str = "../saved_models/mm_retrieval",
        query_encoder_save_dir: str = "query_encoder",
        passage_encoder_save_dir: str = "passage_encoder",
        table_encoder_save_dir: str = "table_encoder",
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
    ):
        """
        Train a TableTextRetrieval model.
        :param data_dir: Directory where training file, dev file and test file are present.
        :param train_filename: Training filename.
        :param dev_filename: Development set filename, file to be used by model in eval step of training.
        :param test_filename: Test set filename, file to be used by model in test step after training.
        :param max_samples: Maximum number of input samples to convert. Can be used for debugging a smaller dataset.
        :param max_processes: The maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None.
        :param batch_size: Total number of samples in 1 batch of data.
        :param embed_meta_fields: Concatenate meta fields with each passage and table.
                                  The default setting in official MMRetrieval embeds page title,
                                  section title and caption with the corresponding table and title with
                                  corresponding text passage.
        :param num_hard_negatives: Number of hard negative passages (passages which are
                                   very similar (high score by BM25) to query but do not contain the answer)-
        :param num_positives: Number of positive passages.
        :param n_epochs: Number of epochs to train the model on.
        :param evaluate_every: Number of training steps after evaluation is run.
        :param n_gpu: Number of gpus to train on.
        :param learning_rate: Learning rate of optimizer.
        :param epsilon: Epsilon parameter of optimizer.
        :param weight_decay: Weight decay parameter of optimizer.
        :param grad_acc_steps: Number of steps to accumulate gradient over before back-propagation is done.
        :param use_amp: Whether to use automatic mixed precision (AMP) or not. The options are:
                    "O0" (FP32)
                    "O1" (Mixed Precision)
                    "O2" (Almost FP16)
                    "O3" (Pure FP16).
                    For more information, refer to: https://nvidia.github.io/apex/amp.html
        :param optimizer_name: What optimizer to use (default: TransformersAdamW).
        :param num_warmup_steps: Number of warmup steps.
        :param optimizer_correct_bias: Whether to correct bias in optimizer.
        :param save_dir: Directory where models are saved.
        :param query_encoder_save_dir: Directory inside save_dir where query_encoder model files are saved.
        :param passage_encoder_save_dir: Directory inside save_dir where passage_encoder model files are saved.
        :param table_encoder_save_dir: Directory inside save_dir where table_encoder model files are saved.
        """

        self.processor.embed_meta_fields = embed_meta_fields
        self.processor.data_dir = Path(data_dir)
        self.processor.train_filename = train_filename
        self.processor.dev_filename = dev_filename
        self.processor.test_filename = test_filename
        self.processor.max_samples = max_samples
        self.processor.dev_split = dev_split
        self.processor.num_hard_negatives = num_hard_negatives
        self.processor.num_positives = num_positives

        if isinstance(self.model, DataParallel):
            self.model.module.connect_heads_with_processor(self.processor.tasks, require_labels=True)
        else:
            self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

        data_silo = DataSilo(
            processor=self.processor, batch_size=batch_size, distributed=False, max_processes=max_processes
        )

        # 5. Create an optimizer
        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={
                "name": optimizer_name,
                "correct_bias": optimizer_correct_bias,
                "weight_decay": weight_decay,
                "eps": epsilon,
            },
            schedule_opts={"name": "LinearWarmup", "num_warmup_steps": num_warmup_steps},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            grad_acc_steps=grad_acc_steps,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
        )

        # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer.create_or_load_checkpoint(
            model=self.model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
            checkpoint_root_dir=Path(checkpoint_root_dir),
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
        )

        # 7. Let it grow! Watch the tracked metrics live on experiment tracker (e.g. Mlflow)
        trainer.train()

        self.model.save(
            Path(save_dir),
            lm1_name=query_encoder_save_dir,
            lm2_name=passage_encoder_save_dir,
            lm3_name=table_encoder_save_dir,
        )
        self.query_tokenizer.save_pretrained(f"{save_dir}/{query_encoder_save_dir}")
        self.passage_tokenizer.save_pretrained(f"{save_dir}/{passage_encoder_save_dir}")
        self.table_tokenizer.save_pretrained(f"{save_dir}/{table_encoder_save_dir}")

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)

    def save(
        self,
        save_dir: Union[Path, str],
        query_encoder_dir: str = "query_encoder",
        passage_encoder_dir: str = "passage_encoder",
        table_encoder_dir: str = "table_encoder",
    ):
        """
        Save TableTextRetriever to the specified directory.

        :param save_dir: Directory to save to.
        :param query_encoder_dir: Directory in save_dir that contains query encoder model.
        :param passage_encoder_dir: Directory in save_dir that contains passage encoder model.
        :param table_encoder_dir: Directory in save_dir that contains table encoder model.
        :return: None
        """
        save_dir = Path(save_dir)
        self.model.save(save_dir, lm1_name=query_encoder_dir, lm2_name=passage_encoder_dir, lm3_name=table_encoder_dir)
        save_dir = str(save_dir)
        self.query_tokenizer.save_pretrained(save_dir + f"/{query_encoder_dir}")
        self.passage_tokenizer.save_pretrained(save_dir + f"/{passage_encoder_dir}")
        self.table_tokenizer.save_pretrained(save_dir + f"/{table_encoder_dir}")

    @classmethod
    def load(
        cls,
        load_dir: Union[Path, str],
        document_store: BaseDocumentStore,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        max_seq_len_table: int = 256,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name", "section_title", "caption"],
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        query_encoder_dir: str = "query_encoder",
        passage_encoder_dir: str = "passage_encoder",
        table_encoder_dir: str = "table_encoder",
        infer_tokenizer_classes: bool = False,
    ):
        """
        Load TableTextRetriever from the specified directory.
        """

        load_dir = Path(load_dir)
        mm_retriever = cls(
            document_store=document_store,
            query_embedding_model=Path(load_dir) / query_encoder_dir,
            passage_embedding_model=Path(load_dir) / passage_encoder_dir,
            table_embedding_model=Path(load_dir) / table_encoder_dir,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            max_seq_len_table=max_seq_len_table,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_meta_fields=embed_meta_fields,
            use_fast_tokenizers=use_fast_tokenizers,
            similarity_function=similarity_function,
            infer_tokenizer_classes=infer_tokenizer_classes,
        )
        logger.info(f"TableTextRetriever model loaded from {load_dir}")

        return mm_retriever


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: str,
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
        max_seq_len: int = 512,
        model_format: Optional[str] = None,
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
        embed_meta_fields: List[str] = [],
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to encode at once.
        :param max_seq_len: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
        :param model_format: Name of framework that was used for saving the model or model type. If no model_format is
                             provided, it will be inferred automatically from the model configuration files.
                             Options:

                             - ``'farm'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
                             - ``'transformers'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
                             - ``'sentence_transformers'`` (will use `_SentenceTransformersEmbeddingEncoder` as embedding encoder)
                             - ``'retribert'`` (will use `_RetribertEmbeddingEncoder` as embedding encoder)
        :param pooling_strategy: Strategy for combining the embeddings from the model (for farm / transformers models only).
                                 Options:

                                 - ``'cls_token'`` (sentence vector)
                                 - ``'reduce_mean'`` (sentence vector)
                                 - ``'reduce_max'`` (sentence vector)
                                 - ``'per_token'`` (individual token vectors)
        :param emb_extraction_layer: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
                                     Default: -1 (very last layer).
        :param top_k: How many documents to return per query.
        :param progress_bar: If true displays progress bar during embedding.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / table to a text pair that is
                                  then used to create the embedding.
                                  This approach is also used in the TableTextRetriever paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        """
        super().__init__()

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.embedding_model = embedding_model
        self.model_version = model_version
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.use_auth_token = use_auth_token
        self.scale_score = scale_score
        self.model_format = self._infer_model_format(embedding_model) if model_format is None else model_format

        logger.info(f"Init retriever using embeddings of model {embedding_model}")

        if self.model_format not in _EMBEDDING_ENCODERS.keys():
            raise ValueError(f"Unknown retriever embedding model format {model_format}")

        if (
            self.embedding_model.startswith("sentence-transformers")
            and model_format
            and model_format != "sentence_transformers"
        ):
            logger.warning(
                f"You seem to be using a Sentence Transformer embedding model but 'model_format' is set to '{self.model_format}'."
                f" You may need to set model_format='sentence_transformers' to ensure correct loading of model."
                f"As an alternative, you can let Haystack derive the format automatically by not setting the "
                f"'model_format' parameter at all."
            )

        self.embedding_encoder = _EMBEDDING_ENCODERS[self.model_format](self)
        self.embed_meta_fields = embed_meta_fields

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index, headers=headers, scale_score=scale_score
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve_batch() since EmbeddingRetriever initialized with document_store=None"
            )
            return [[] * len(queries)]  # type: ignore

        documents = []
        query_embs = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(texts=batch))
        for query_emb, cur_filters in zip(query_embs, filters):
            cur_docs = self.document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index,
                headers=headers,
                scale_score=scale_score,
            )
            documents.append(cur_docs)

        return documents

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries.

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        # for backward compatibility: cast pure str input
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        return self.embedding_encoder.embed_queries(texts)

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed
        :return: Embeddings, one per input document
        """
        docs = self._preprocess_documents(docs)
        return self.embedding_encoder.embed_documents(docs)

    def _preprocess_documents(self, docs: List[Document]) -> List[Document]:
        """
        Turns table documents into text documents by representing the table in csv format.
        This allows us to use text embedding models for table retrieval.
        It also concatenates specified meta data fields with the text representations.

        :param docs: List of documents to linearize. If the document is not a table, it is returned as is.
        :return: List of documents with meta data + linearized tables or original documents if they are not tables.
        """
        linearized_docs = []
        for doc in docs:
            doc = deepcopy(doc)
            if doc.content_type == "table":
                if isinstance(doc.content, pd.DataFrame):
                    doc.content = doc.content.to_csv(index=False)
                else:
                    raise HaystackError("Documents of type 'table' need to have a pd.DataFrame as content field")
            meta_data_fields = [doc.meta[key] for key in self.embed_meta_fields if key in doc.meta and doc.meta[key]]
            doc.content = "\n".join(meta_data_fields + [doc.content])
            linearized_docs.append(doc)
        return linearized_docs

    @staticmethod
    def _infer_model_format(model_name_or_path: str) -> str:
        # Check if model name is a local directory with sentence transformers config file in it
        if Path(model_name_or_path).exists():
            if Path(f"{model_name_or_path}/config_sentence_transformers.json").exists():
                return "sentence_transformers"
        # Check if sentence transformers config file in model hub
        else:
            try:
                hf_hub_download(repo_id=model_name_or_path, filename="config_sentence_transformers.json")
                return "sentence_transformers"
            except HTTPError:
                pass

        # Check if retribert model
        config = AutoConfig.from_pretrained(model_name_or_path)
        if config.model_type == "retribert":
            return "retribert"

        # Model is neither sentence-transformers nor retribert model -> use _DefaultEmbeddingEncoder
        return "farm"

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: int = None,
        batch_size: int = 16,
    ) -> None:
        """
        Trains/adapts the underlying embedding model.

        Each training data example is a dictionary with the following keys:

        * question: the question string
        * pos_doc: the positive document string
        * neg_doc: the negative document string
        * score: the score margin


        :param training_data: The training data
        :type training_data: List[Dict[str, Any]]
        :param learning_rate: The learning rate
        :type learning_rate: float
        :param n_epochs: The number of epochs
        :type n_epochs: int
        :param num_warmup_steps: The number of warmup steps
        :type num_warmup_steps: int
        :param batch_size: The batch size to use for the training, defaults to 16
        :type batch_size: int (optional)
        """
        self.embedding_encoder.train(
            training_data,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            num_warmup_steps=num_warmup_steps,
            batch_size=batch_size,
        )

    def save(self, save_dir: Union[Path, str]) -> None:
        """
        Save the model to the given directory

        :param save_dir: The directory where the model will be saved
        :type save_dir: Union[Path, str]
        """
        self.embedding_encoder.save(save_dir=save_dir)
