import logging
from typing import List, Union, Tuple, Optional
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from haystack.document_store.base import BaseDocumentStore
from haystack import Document
from haystack.retriever.base import BaseRetriever

from farm.infer import Inferencer
from farm.modeling.tokenization import Tokenizer
from farm.modeling.language_model import LanguageModel
from farm.modeling.biadaptive_model import BiAdaptiveModel
from farm.modeling.prediction_head import TextSimilarityHead
from farm.data_handler.processor import TextSimilarityProcessor
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
                 max_seq_len_query: int = 64,
                 max_seq_len_passage: int = 256,
                 use_gpu: bool = True,
                 batch_size: int = 16,
                 embed_title: bool = True,
                 use_fast_tokenizers: bool = True,
                 similarity_function: str = "dot_product"
                 ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format

        **Example:**

                ```python
                # remote model from FAIR
                DensePassageRetriever(document_store=your_doc_store,	
                >                    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",	
                >                    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")	
                # or from local path
                DensePassageRetriever(document_store=your_doc_store,	
                >                    query_embedding_model="model_directory/question-encoder",	
                >                   passage_embedding_model="model_directory/context-encoder")
                ```

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models
                                      Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by hugging-face transformers' modelhub models
                                        Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param use_gpu: Whether to use gpu or not
        :param batch_size: Number of questions or passages to encode at once
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        """

        self.document_store = document_store
        self.batch_size = batch_size
        self.max_seq_len_passage = max_seq_len_passage
        self.max_seq_len_query = max_seq_len_query

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.embed_title = embed_title

        # Init & Load Encoders
        self.query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=query_embedding_model,
                                              do_lower_case=True, use_fast=use_fast_tokenizers)
        self.query_encoder = LanguageModel.load(pretrained_model_name_or_path=query_embedding_model,
                                                language_model_class="DPRQuestionEncoder")

        self.passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=passage_embedding_model,
                                                do_lower_case=True, use_fast=use_fast_tokenizers)
        self.passage_encoder = LanguageModel.load(pretrained_model_name_or_path=passage_embedding_model,
                                                  language_model_class="DPRContextEncoder")

        self.processor = TextSimilarityProcessor(tokenizer=self.query_tokenizer,
                                                 passage_tokenizer=self.passage_tokenizer,
                                                 max_seq_len_passage=self.max_seq_len_passage,
                                                 max_seq_len_query=self.max_seq_len_query,
                                                 label_list=["hard_negative", "positive"],
                                                 metric="text_similarity_metric",
                                                 embed_title=self.embed_title,
                                                 num_hard_negatives=0,
                                                 num_negatives=0)

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

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            index = self.document_store.index
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k, filters=filters, index=index)
        return documents

    def _get_predictions(self, dicts, tokenizer):
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
        dataset, tensor_names, baskets = self.processor.dataset_from_dicts(
            dicts, indices=[i for i in range(len(dicts))], return_baskets=True
        )

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        all_embeddings = {"query": torch.tensor([]).to(self.device), "passages": torch.tensor([]).to(self.device)}
        self.model.eval()
        for i, batch in enumerate(tqdm(data_loader, desc=f"Inferencing Samples", unit=" Batches", disable=False)):
            batch = {key: batch[key].to(self.device) for key in batch}

            # get logits
            with torch.no_grad():
                query_embeddings, passage_embeddings = self.model.forward(**batch)[0]
                all_embeddings["query"] = torch.cat((all_embeddings["query"], query_embeddings), dim=0) \
                                                     if isinstance(query_embeddings, torch.Tensor) else None
                all_embeddings["passages"] = torch.cat((all_embeddings["passages"], passage_embeddings), dim=0) \
                                                    if isinstance(passage_embeddings, torch.Tensor) else None

        # convert embeddings to numpy array
        for k, v in all_embeddings.items():
            all_embeddings[k] = v.cpu().numpy() if v!=None else None
        return all_embeddings

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [{'query': q} for q in texts]
        result = self._get_predictions(queries, self.query_tokenizer)["query"]
        return result

    def embed_passages(self, docs: List[Document]) -> List[np.array]:
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
        embeddings = self._get_predictions(passages, self.passage_tokenizer)["passages"]

        return embeddings

    def train(self,
              data_dir: str,
              train_filename: str,
              dev_filename: str = None,
              test_filename: str = None,
              batch_size: int = 2,
              embed_title: bool = True,
              num_hard_negatives: int = 1,
              num_negatives: int = 0,
              n_epochs: int = 3,
              evaluate_every: int = 1000,
              n_gpu: int = 1,
              learning_rate: float = 1e-5,
              epsilon: float = 1e-08,
              weight_decay: float = 0.0,
              num_warmup_steps: int = 100,
              grad_acc_steps: int = 1,
              optimizer_name: str = "TransformersAdamW",
              optimizer_correct_bias: bool = True,
              save_dir: str = "../saved_models/dpr-tutorial",
              query_encoder_save_dir: str = "lm1",
              passage_encoder_save_dir: str = "lm2"
              ):
        """
        train a DensePassageRetrieval model
        :param data_dir: Directory where training file, dev file and test file are present
        :param train_filename: training filename
        :param dev_filename: development set filename, file to be used by model in eval step of training
        :param test_filename: test set filename, file to be used by model in test step after training
        :param batch_size: total number of samples in 1 batch of data
        :param embed_title: whether to concatenate passage title with each passage. The default setting in official DPR embeds passage title with the corresponding passage
        :param num_hard_negatives: number of hard negative passages(passages which are very similar(high score by BM25) to query but do not contain the answer
        :param num_negatives: number of negative passages(any random passage from dataset which do not contain answer to query)
        :param n_epochs: number of epochs to train the model on
        :param evaluate_every: number of training steps after evaluation is run
        :param n_gpu: number of gpus to train on
        :param learning_rate: learning rate of optimizer
        :param epsilon: epsilon parameter of optimizer
        :param weight_decay: weight decay parameter of optimizer
        :param grad_acc_steps: number of steps to accumulate gradient over before back-propagation is done
        :param optimizer_name: what optimizer to use (default: TransformersAdamW)
        :param num_warmup_steps: number of warmup steps
        :param optimizer_correct_bias: Whether to correct bias in optimizer
        :param save_dir: directory where models are saved
        :param query_encoder_save_dir: directory inside save_dir where query_encoder model files are saved
        :param passage_encoder_save_dir: directory inside save_dir where passage_encoder model files are saved
        """

        self.embed_title = embed_title
        self.processor = TextSimilarityProcessor(tokenizer=self.query_tokenizer,
                                                 passage_tokenizer=self.passage_tokenizer,
                                                 max_seq_len_passage=self.max_seq_len_passage,
                                                 max_seq_len_query=self.max_seq_len_query,
                                                 label_list=["hard_negative", "positive"],
                                                 metric="text_similarity_metric",
                                                 data_dir=data_dir,
                                                 train_filename=train_filename,
                                                 dev_filename=dev_filename,
                                                 test_filename=test_filename,
                                                 embed_title=self.embed_title,
                                                 num_hard_negatives=num_hard_negatives,
                                                 num_negatives=num_negatives)

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

        data_silo = DataSilo(processor=self.processor, batch_size=batch_size, distributed=False)

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
            device=self.device
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
        )

        # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
        trainer.train()

        self.model.save(Path(save_dir), lm1_name=query_encoder_save_dir, lm2_name=passage_encoder_save_dir)
        self.processor.save(Path(save_dir))

    def save(self, save_dir: Union[Path, str]):
        save_dir = Path(save_dir)
        self.model.save(save_dir, lm1_name="query_encoder", lm2_name="passage_encoder")
        save_dir = str(save_dir)
        self.query_tokenizer.save_pretrained(save_dir + "/query_encoder")
        self.passage_tokenizer.save_pretrained(save_dir + "/passage_encoder")

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
             ):

        load_dir = Path(load_dir)
        dpr = cls(
            document_store=document_store,
            query_embedding_model=Path(load_dir) / "query_encoder",
            passage_embedding_model=Path(load_dir) / "passage_encoder",
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers,
            similarity_function=similarity_function
        )

        return dpr


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: str,
        use_gpu: bool = True,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'deepset/sentence_bert'``
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
        """
        self.document_store = document_store
        self.model_format = model_format
        self.embedding_model = embedding_model
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer

        logger.info(f"Init retriever using embeddings of model {embedding_model}")
        if model_format == "farm" or model_format == "transformers":
            self.embedding_model = Inferencer.load(
                embedding_model, task_type="embeddings", extraction_strategy=self.pooling_strategy,
                extraction_layer=self.emb_extraction_layer, gpu=use_gpu, batch_size=4, max_seq_len=512, num_processes=0
            )

        elif model_format == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Can't find package `sentence-transformers` \n"
                                  "You can install it via `pip install sentence-transformers` \n"
                                  "For details see https://github.com/UKPLab/sentence-transformers ")
            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            if use_gpu:
                device = "cuda"
            else:
                device = "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        else:
            raise NotImplementedError

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            index = self.document_store.index
        query_emb = self.embed(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], filters=filters,
                                                           top_k=top_k, index=index)
        return documents

    def embed(self, texts: Union[List[str], str]) -> List[np.array]:
        """
        Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)

        :param texts: Texts to embed
        :return: List of embeddings (one per input text). Each embedding is a list of floats.
        """

        # for backward compatibility: cast pure str input
        if type(texts) == str:
            texts = [texts]  # type: ignore
        assert type(texts) == list, "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        if self.model_format == "farm" or self.model_format == "transformers":
            # TODO: FARM's `sample_to_features_text` need to fix following warning -
            # tokenization_utils.py:460: FutureWarning: `is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.
            emb = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])  # type: ignore
            emb = [(r["vec"]) for r in emb]
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            # get back list of numpy embedding vectors
            emb = self.embedding_model.encode(texts)  # type: ignore
            emb = [r for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries. For this Retriever type: The same as calling .embed()

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        return self.embed(texts)

    def embed_passages(self, docs: List[Document]) -> List[np.array]:
        """
        Create embeddings for a list of passages. For this Retriever type: The same as calling .embed()

        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.text for d in docs]

        return self.embed(texts)
