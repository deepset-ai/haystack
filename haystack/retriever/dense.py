import logging
from abc import abstractmethod
from typing import List, Union, Optional
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from haystack.document_store.base import BaseDocumentStore
from haystack import Document
from haystack.retriever.base import BaseRetriever
from haystack.utils import get_device

from farm.infer import Inferencer
from farm.modeling.tokenization import Tokenizer
from farm.modeling.language_model import LanguageModel
from farm.modeling.biadaptive_model import BiAdaptiveModel
from farm.modeling.prediction_head import TextSimilarityHead
from farm.data_handler.processor import TextSimilarityProcessor, InferenceProcessor
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.dataloader import NamedDataLoader
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from torch.utils.data.sampler import SequentialSampler


logger = logging.getLogger(__name__)


class DensePassageRetriever(BaseRetriever):
    """
        Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
        See the original paper for more details:
        Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
        (https://arxiv.org/abs/2004.04906).
    """

    def __init__(self,
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
                 progress_bar: bool = True
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
        :param use_gpu: Whether to use gpu or not
        :param batch_size: Number of questions or passages to encode at once
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
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            document_store=document_store, query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
            model_version=model_version, max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
            top_k=top_k, use_gpu=use_gpu, batch_size=batch_size, embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers, infer_tokenizer_classes=infer_tokenizer_classes,
            similarity_function=similarity_function, progress_bar=progress_bar,
        )

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k

        if document_store is None:
           logger.warning("DensePassageRetriever initialized without a document store. "
                          "This is fine if you are performing DPR training. "
                          "Otherwise, please provide a document store in the constructor.")
        elif document_store.similarity != "dot_product":
            logger.warning(f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                           "We recommend you use dot_product instead. "
                           "This can be set when initializing the DocumentStore")

        self.device = get_device(use_gpu)

        self.infer_tokenizer_classes = infer_tokenizer_classes
        tokenizers_default_classes = {
            "query": "DPRQuestionEncoderTokenizer",
            "passage": "DPRContextEncoderTokenizer"
        }
        if self.infer_tokenizer_classes:
            tokenizers_default_classes["query"] = None   # type: ignore
            tokenizers_default_classes["passage"] = None # type: ignore

        # Init & Load Encoders
        self.query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=query_embedding_model,
                                              revision=model_version,
                                              do_lower_case=True,
                                              use_fast=use_fast_tokenizers,
                                              tokenizer_class=tokenizers_default_classes["query"])
        self.query_encoder = LanguageModel.load(pretrained_model_name_or_path=query_embedding_model,
                                                revision=model_version,
                                                language_model_class="DPRQuestionEncoder")
        self.passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=passage_embedding_model,
                                                revision=model_version,
                                                do_lower_case=True,
                                                use_fast=use_fast_tokenizers,
                                                tokenizer_class=tokenizers_default_classes["passage"])
        self.passage_encoder = LanguageModel.load(pretrained_model_name_or_path=passage_embedding_model,
                                                  revision=model_version,
                                                  language_model_class="DPRContextEncoder")

        self.processor = TextSimilarityProcessor(query_tokenizer=self.query_tokenizer,
                                                 passage_tokenizer=self.passage_tokenizer,
                                                 max_seq_len_passage=max_seq_len_passage,
                                                 max_seq_len_query=max_seq_len_query,
                                                 label_list=["hard_negative", "positive"],
                                                 metric="text_similarity_metric",
                                                 embed_title=embed_title,
                                                 num_hard_negatives=0,
                                                 num_positives=1)
        prediction_head = TextSimilarityHead(similarity_function=similarity_function)
        self.model = BiAdaptiveModel(
            language_model1=self.query_encoder,
            language_model2=self.passage_encoder,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm1_output_types=["per_sequence"],
            lm2_output_types=["per_sequence"],
            device=self.device,
        )

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k, filters=filters, index=index)
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
            disable_tqdm=True
        else:
            disable_tqdm = not self.progress_bar

        with tqdm(total=len(data_loader)*self.batch_size, unit=" Docs", desc=f"Create embeddings", position=1,
                  leave=False, disable=disable_tqdm) as progress_bar:
            for batch in data_loader:
                batch = {key: batch[key].to(self.device) for key in batch}

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
        queries = [{'query': q} for q in texts]
        result = self._get_predictions(queries)["query"]
        return result

    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of passages using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within Haystack.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        passages = [{'passages': [{
            "title": d.meta["name"] if d.meta and "name" in d.meta else "",
            "text": d.text,
            "label": d.meta["label"] if d.meta and "label" in d.meta else "positive",
            "external_id": d.id}]
        } for d in docs]
        embeddings = self._get_predictions(passages)["passages"]

        return embeddings

    def train(self,
              data_dir: str,
              train_filename: str,
              dev_filename: str = None,
              test_filename: str = None,
              max_sample: int = None,
              max_processes: int = 128,
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
              optimizer_name: str = "TransformersAdamW",
              optimizer_correct_bias: bool = True,
              save_dir: str = "../saved_models/dpr",
              query_encoder_save_dir: str = "query_encoder",
              passage_encoder_save_dir: str = "passage_encoder"
              ):
        """
        train a DensePassageRetrieval model
        :param data_dir: Directory where training file, dev file and test file are present
        :param train_filename: training filename
        :param dev_filename: development set filename, file to be used by model in eval step of training
        :param test_filename: test set filename, file to be used by model in test step after training
        :param max_sample: maximum number of input samples to convert. Can be used for debugging a smaller dataset.
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
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
        :param optimizer_name: what optimizer to use (default: TransformersAdamW)
        :param num_warmup_steps: number of warmup steps
        :param optimizer_correct_bias: Whether to correct bias in optimizer
        :param save_dir: directory where models are saved
        :param query_encoder_save_dir: directory inside save_dir where query_encoder model files are saved
        :param passage_encoder_save_dir: directory inside save_dir where passage_encoder model files are saved
        """

        self.processor.embed_title = embed_title
        self.processor.data_dir = Path(data_dir)
        self.processor.train_filename = train_filename
        self.processor.dev_filename = dev_filename
        self.processor.test_filename = test_filename
        self.processor.max_sample = max_sample
        self.processor.dev_split = dev_split
        self.processor.num_hard_negatives = num_hard_negatives
        self.processor.num_positives = num_positives

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

        data_silo = DataSilo(processor=self.processor, batch_size=batch_size, distributed=False, max_processes=max_processes)

        # 5. Create an optimizer
        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={"name": optimizer_name, "correct_bias": optimizer_correct_bias,
                            "weight_decay": weight_decay, "eps": epsilon},
            schedule_opts={"name": "LinearWarmup", "num_warmup_steps": num_warmup_steps},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            grad_acc_steps=grad_acc_steps,
            device=self.device,
            use_amp=use_amp
        )

        # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=self.device,
            use_amp=use_amp
        )

        # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
        trainer.train()

        self.model.save(Path(save_dir), lm1_name=query_encoder_save_dir, lm2_name=passage_encoder_save_dir)
        self.query_tokenizer.save_pretrained(f"{save_dir}/{query_encoder_save_dir}")
        self.passage_tokenizer.save_pretrained(f"{save_dir}/{passage_encoder_save_dir}")

    def save(self, save_dir: Union[Path, str], query_encoder_dir: str = "query_encoder",
             passage_encoder_dir: str = "passage_encoder"):
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
    def load(cls,
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
             infer_tokenizer_classes: bool = False
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
            infer_tokenizer_classes=infer_tokenizer_classes
        )
        logger.info(f"DPR model loaded from {load_dir}")

        return dpr


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: str,
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'deepset/sentence_bert'``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use gpu or not
        :param model_format: Name of framework that was used for saving the model. Options:

                             - ``'farm'``
                             - ``'transformers'``
                             - ``'sentence_transformers'``
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
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            document_store=document_store, embedding_model=embedding_model, model_version=model_version,
            use_gpu=use_gpu, model_format=model_format, pooling_strategy=pooling_strategy,
            emb_extraction_layer=emb_extraction_layer, top_k=top_k,
        )

        self.document_store = document_store
        self.embedding_model = embedding_model
        self.model_format = model_format
        self.model_version = model_version
        self.use_gpu = use_gpu
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar

        logger.info(f"Init retriever using embeddings of model {embedding_model}")
        self.embedding_encoder = _EmbeddingEncoderFactory.get_embedding_retriever_impl(self, model_format)

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], filters=filters,
                                                           top_k=top_k, index=index)
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

    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of passages.

        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        return self.embedding_encoder.embed_passages(docs)


class _EmbeddingEncoder:

    @abstractmethod
    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries.

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        pass

    @abstractmethod
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of passages.

        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        pass


class _EmbeddingEncoderFactory:

    @staticmethod
    def get_embedding_retriever_impl(retriever: EmbeddingRetriever, model_format: str) -> _EmbeddingEncoder:
        if model_format == "farm" or model_format == "transformers":
            return _DefaultEmbeddingEncoder(retriever)
        elif model_format == "sentence_transformers":
            return _SentenceTransformersEmbeddingEncoder(retriever)
        elif model_format == "retribert":
            return _RetribertEmbeddingEncoder(retriever)
        else:
            raise ValueError(f"Unknown retriever embedding model format {model_format}")


class _DefaultEmbeddingEncoder(_EmbeddingEncoder):

    def __init__(
            self,
            retriever: EmbeddingRetriever
    ):

        self.embedding_model = Inferencer.load(
            retriever.embedding_model, revision=retriever.model_version, task_type="embeddings",
            extraction_strategy=retriever.pooling_strategy,
            extraction_layer=retriever.emb_extraction_layer, gpu=retriever.use_gpu,
            batch_size=4, max_seq_len=512, num_processes=0
        )
        # Check that document_store has the right similarity function
        similarity = retriever.document_store.similarity
        # If we are using a sentence transformer model
        if "sentence" in retriever.embedding_model.lower() and similarity != "cosine":
            logger.warning(f"You seem to be using a Sentence Transformer with the {similarity} function. "
                           f"We recommend using cosine instead. "
                           f"This can be set when initializing the DocumentStore")
        elif "dpr" in retriever.embedding_model.lower() and similarity != "dot_product":
            logger.warning(f"You seem to be using a DPR model with the {similarity} function. "
                           f"We recommend using dot_product instead. "
                           f"This can be set when initializing the DocumentStore")

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[np.ndarray]:
        # TODO: FARM's `sample_to_features_text` need to fix following warning -
        # tokenization_utils.py:460: FutureWarning: `is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.
        emb = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])
        emb = [(r["vec"]) for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self.embed(texts)

    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        passages = [d.text for d in docs] # type: ignore
        return self.embed(passages)


class _SentenceTransformersEmbeddingEncoder(_EmbeddingEncoder):

    def __init__(
            self,
            retriever: EmbeddingRetriever
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Can't find package `sentence-transformers` \n"
                              "You can install it via `pip install sentence-transformers` \n"
                              "For details see https://github.com/UKPLab/sentence-transformers ")
        # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
        # e.g. 'roberta-base-nli-stsb-mean-tokens'
        device = get_device(retriever.use_gpu)
        self.embedding_model = SentenceTransformer(retriever.embedding_model, device=device)
        self.show_progress_bar = retriever.progress_bar
        document_store = retriever.document_store
        if document_store.similarity != "cosine":
            logger.warning(
                f"You are using a Sentence Transformer with the {document_store.similarity} function. "
                f"We recommend using cosine instead. "
                f"This can be set when initializing the DocumentStore")

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[np.ndarray]:
        # texts can be a list of strings or a list of [title, text]
        # get back list of numpy embedding vectors
        emb = self.embedding_model.encode(texts, batch_size=200, show_progress_bar=self.show_progress_bar)
        emb = [r for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self.embed(texts)

    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.text] for d in docs]  # type: ignore
        return self.embed(passages)


class _RetribertEmbeddingEncoder(_EmbeddingEncoder):

    def __init__(
            self,
            retriever: EmbeddingRetriever
    ):

        self.progress_bar = retriever.progress_bar
        if retriever.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        embedding_tokenizer = AutoTokenizer.from_pretrained(retriever.embedding_model,
                                                            use_fast_tokenizers=True)
        self.embedding_model = AutoModel.from_pretrained(retriever.embedding_model).to(self.device)
        self.processor = InferenceProcessor(tokenizer=embedding_tokenizer,
                                            max_seq_len=embedding_tokenizer.max_len_single_sentence)

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:

        queries = [{"text": q} for q in texts]
        dataloader = self._create_dataloader(queries)

        embeddings: List[np.ndarray] = []
        disable_tqdm = True if len(dataloader) == 1 else not self.progress_bar

        for i, batch in enumerate(tqdm(dataloader, desc=f"Creating Embeddings", unit=" Batches", disable=disable_tqdm)):
            batch = {key: batch[key].to(self.device) for key in batch}
            with torch.no_grad():
                q_reps = self.embedding_model.embed_questions(input_ids=batch["input_ids"],
                                                              attention_mask=batch["padding_mask"]).cpu().numpy()
            embeddings.append(q_reps)

        return np.concatenate(embeddings)

    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:

        doc_text = [{"text": d.text} for d in docs]
        dataloader = self._create_dataloader(doc_text)

        embeddings: List[np.ndarray] = []
        disable_tqdm = True if len(dataloader) == 1 else not self.progress_bar

        for i, batch in enumerate(tqdm(dataloader, desc=f"Creating Embeddings", unit=" Batches", disable=disable_tqdm)):
            batch = {key: batch[key].to(self.device) for key in batch}
            with torch.no_grad():
                q_reps = self.embedding_model.embed_answers(input_ids=batch["input_ids"],
                                                            attention_mask=batch["padding_mask"]).cpu().numpy()
            embeddings.append(q_reps)

        return np.concatenate(embeddings)

    def _create_dataloader(self, text_to_encode: List[dict]) -> NamedDataLoader:

        dataset, tensor_names, _ = self.processor.dataset_from_dicts(text_to_encode,
                                                                     indices=[i for i in range(len(text_to_encode))])
        dataloader = NamedDataLoader(dataset=dataset, sampler=SequentialSampler(dataset),
                                     batch_size=32, tensor_names=tensor_names)
        return dataloader
