from typing import Iterable, get_args, Union, Optional, Dict, List, Any

import numbers
import logging
from pathlib import Path

import tqdm
import torch
import numpy as np
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import ConcatDataset, TensorDataset

from haystack.nodes.retriever import BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.model.multiadaptive_model import MultiAdaptiveModel
from haystack.modeling.model.embedding_similarity_head import EmbeddingSimilarityHead
from haystack.modeling.data_handler.multimodal_similarity_processor import MultiModalDatasetProcessor
from haystack.modeling.model.language_model import get_language_model
from haystack.modeling.model.feature_extraction import FeatureExtractor
from haystack.errors import NodeError, ModelingError
from haystack.schema import ContentTypes, Document
from haystack.modeling.data_handler.multimodal_samples.text import TextSample


logger = logging.getLogger(__name__)


class MultiModalRetrieverError(NodeError):
    pass


PASSAGE_FROM_DOCS = {
    "text": lambda doc: {"text": doc.content},
    "table": lambda doc: {"columns": doc.content.columns.tolist(), "rows": doc.content.values.tolist()},
    "image": lambda doc: {"image": np.array(Image.open(doc.content).convert("RGB"))},
}


def get_devices(devices: List[Union[str, torch.device]]) -> List[torch.device]:
    """
    Convert a list of device names into a list of Torch devices,
    depending on the system's configuration and hardware.
    """
    if devices is not None:
        return [torch.device(device) for device in devices]
    elif torch.cuda.is_available():
        return [torch.device(device) for device in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def flatten(iterable: Any):
    """
    Flatten an arbitrarily nested list. Does not unpack tuples or other Iterables.
    Yields a generator. Use `list()` to compute the full list.

    >> list(flatten([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten([[1, 2], 3]))
    [1, 2, 3]
    """
    if isinstance(iterable, list):
        for item in iterable:
            yield from flatten(item)
    else:
        yield (iterable)


class _EvaluationMixin:
    pass


class _TrainingMixin:
    pass


class MultiModalRetriever(BaseRetriever, _EvaluationMixin, _TrainingMixin):
    """
    Retriever that uses a multiple encoder to jointly retrieve among a database consisting of different
    data types. See the original paper for more details:
    Kostić, Bogdan, et al. (2021): "Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models"
    (https://arxiv.org/abs/2108.04049),
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "facebook/data2vec-text-base",
        passage_embedding_models: Dict[ContentTypes, Union[Path, str]] = {"text": "facebook/data2vec-text-base"},
        query_feature_extractor_params: Dict[str, Any] = None,
        passage_feature_extractors_params: Dict[str, Dict[str, Any]] = None,
        top_k: int = 10,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name"],
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        Init the Retriever and all its models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format.

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models.
        :param passage_embedding_models: Dictionary matching a local path or remote name of passage encoder checkpoint with
            the content type it should handle ("text", "table", "image", etc...).
            The format equals the one used by hugging-face transformers' modelhub models.
        :param max_seq_len_query:Longest length of each passage/context sequence. Represents the maximum number of tokens for the passage text.
            Longer ones will be cut down.
        :param max_seq_len_passages: Dictionary matching the longest length of each query sequence with the content_type they refer to.
            Represents the maximum number of tokens. Longer ones will be cut down.
        :param top_k: How many documents to return per query.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / image to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/simage/tensor_attributes.html?highlight=torch%20device#torch.torch.device
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

        self.devices = get_devices(devices)
        if batch_size < len(self.devices):
            logger.warning("Batch size is lower than the number of devices. Not all GPUs will be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.embed_meta_fields = embed_meta_fields
        self.scale_score = scale_score

        self.query_feature_extractor_params = query_feature_extractor_params

        self.passage_feature_extractors_params = {
            content_type: {
                "max_length": 256,
                # "add_special_tokens": True,
                # "truncation": True,
                # "truncation_strategy": "longest_first",
                # "padding": "max_length",
                # "return_token_type_ids": True
            }
            | (passage_feature_extractors_params or {}).get(content_type, {})
            for content_type in get_args(ContentTypes)
        }

        if document_store is None:
            raise MultiModalRetrieverError("Please provide a DocumentStore instance to the Retriever.")
            # logger.warning(
            #     "MultiModalRetriever initialized without a document store. "
            #     "This is fine if you are training. "
            #     "Otherwise, please provide a document store in the constructor."
            # )

        # Language Models
        self.query_feature_extractor = FeatureExtractor(
            pretrained_model_name_or_path=query_embedding_model, do_lower_case=True, use_auth_token=use_auth_token
        )
        self.query_model = get_language_model(
            pretrained_model_name_or_path=query_embedding_model, use_auth_token=use_auth_token
        )

        self.passage_feature_extractors = {}
        self.passage_models = {}
        for content_type, embedding_model in passage_embedding_models.items():
            self.passage_feature_extractors[content_type] = FeatureExtractor(
                pretrained_model_name_or_path=embedding_model, do_lower_case=True, use_auth_token=use_auth_token
            )
            self.passage_models[content_type] = get_language_model(
                pretrained_model_name_or_path=embedding_model, use_auth_token=use_auth_token
            )

        prediction_head = EmbeddingSimilarityHead(
            similarity_function=similarity_function, global_loss_buffer_size=global_loss_buffer_size
        )

        self.model = MultiAdaptiveModel(
            query_model=self.query_model,
            context_models=self.passage_models,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            query_output_types=["per_sequence"],
            context_output_types=["per_sequence"],
            device=self.devices[0],
        )

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)

        self.processor = MultiModalDatasetProcessor(
            query_feature_extractor=self.query_feature_extractor,
            passage_feature_extractors=self.passage_feature_extractors,
            # max_seq_len_query=max_seq_len_query,
            # max_seq_len_passages=max_seq_len_passages,
            # label_list=["hard_negative", "positive"],
            # metric="text_similarity_metric",
            # embed_meta_fields=embed_meta_fields,
            # num_hard_negatives=0,
            # num_positives=1,
        )

        # self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ) -> List[Document]:
        return self.retrieve_batch(
            queries=[query],
            filters=[filters],
            top_k=top_k,
            index=index,
            headers=headers,
            scale_score=scale_score,
            batch_size=1,
        )[0]

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

        Returns a list of lists of Documents (one list per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).
        :param top_k: How many documents to return per query. Must be > 0
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time. Must be > 0
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        # TODO Support non-textual queries, then add the following disclaimer to the docstring:
        #
        # "This method assumes all queries are of the same data type. Mixed-type query batches (i.e. one image and one text)
        # are currently not supported. Please group the queries by type and call `retrieve()` on uniform batches only."

        if not isinstance(filters, list):
            filters = [filters or {}] * len(queries)

        elif len(filters) != len(queries):
            raise MultiModalRetrieverError(
                "Number of filters does not match number of queries. Please provide as many filters"
                " as queries or a single filter that will be applied to each query."
            )

        top_k = top_k or self.top_k
        batch_size = batch_size or self.batch_size or len(queries)
        index = index or self.document_store.index
        scale_score = scale_score or self.scale_score

        # Create batches of queries
        batched_queries = [
            queries[batch_index : batch_index + batch_size] for batch_index in range(0, len(queries), batch_size)
        ]

        # Embed the queries
        batched_query_embeddings = [self.embed_queries(texts=batch) for batch in batched_queries]

        # Query documents by embedding (the actual retrieval step)
        documents = []
        for query_embedding, query_filters in zip(batched_query_embeddings, filters):
            docs = self.document_store.query_by_embedding(
                query_emb=query_embedding,
                top_k=top_k,
                filters=query_filters,
                index=index,
                headers=headers,
                scale_score=scale_score,
            )
            documents.append(docs)
        return documents

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries using the query encoder.

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        # extract the features
        features: List[Dict[str, Any]] = []
        for query in queries:
            features = TextSample.get_features(
                text=query,
                feature_extractor=self.query_feature_extractor,
                extraction_params=self.query_feature_extractor_params,
            )
            features.extend(features)  # The list is flat on purpose

        if not features:
            raise ModelingError(
                f"Could not extract features from the queries. "
                f"Check that your feature extractor ({self.query_feature_extractor}) matches your query type (str)"
            )

        all_tensors = []
        for tensor_name, tensor in features[0].items():
            # Check whether a non-integer will be silently converted to torch.long
            try:
                if isinstance(tensor, numbers.Number):
                    base = tensor
                elif isinstance(tensor, list):
                    base = list(flatten(tensor))[0]
                else:
                    base = tensor.ravel()[0]

                if not np.issubdtype(type(base), np.integer):
                    logger.warning(
                        f"A non-integer value for feature '{tensor_name}' with value "
                        f"'{base}' will be converted to a torch tensor of dtype long. This is usually an issue."
                    )
            except Exception as e:
                logger.warning(
                    f"Could not determine type for feature '{tensor_name}'. "
                    "Converting now to a tensor of default type long. "
                    f"Original error: {e}"
                )

            # Convert all remaining feature objects to torch.long tensors
            tensors = torch.as_tensor(np.array([feature[tensor_name] for feature in features]), dtype=torch.long)
            all_tensors.append(tensors)

        dataset = TensorDataset(*all_tensors)

        data_loader = NamedDataLoader(
            dataset=dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size,
            tensor_names=list(features[0].keys()),
        )

        # Perform inference on the queries
        all_embeddings = []
        with tqdm(
            total=len(data_loader) * self.batch_size,
            unit=" Docs",
            desc=f"Create embeddings",
            position=1,
            leave=False,
            disable=(dataset and len(dataset) == 1)
            or not self.progress_bar,  # On eval we don't want a progress bar for every query
        ) as progress_bar:

            for batch in data_loader:
                batch = {key: batch[key].to(self.devices[0]) for key in batch}

                # get logits
                with torch.no_grad():
                    query_embeddings, _ = self.model.forward(inputs_by_model={"query": batch})[0]
                    all_embeddings.append(query_embeddings.cpu().numpy())

                progress_bar.update(self.batch_size)

        return np.concatenate(all_embeddings)

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents using the appropriate encoder.

        :param docs: List of Document object to compute embeddings for.
        :return: Embeddings of documents. Shape: (batch_size, embedding_dim)
        """
        if self.processor.num_hard_negatives != 0:
            logger.warning(
                f"'num_hard_negatives' is set to {self.processor.num_hard_negatives}, but inference does "
                f"not require any hard negatives. Setting num_hard_negatives to 0."
            )
            self.processor.num_hard_negatives = 0

        model_input = []
        for doc in docs:
            if doc.content_type not in PASSAGE_FROM_DOCS.keys():
                raise MultiModalRetrieverError(f"Unknown content type '{doc.content_type}'.")

            passage = {
                "passages": [
                    {
                        "meta": [
                            doc.meta[meta_field]
                            for meta_field in self.embed_meta_fields
                            if meta_field in doc.meta and isinstance(doc.meta[meta_field], str)
                        ],
                        "label": doc.meta["label"] if doc.meta and "label" in doc.meta else "positive",
                        "type": doc.content_type,
                        "external_id": doc.id,
                        **PASSAGE_FROM_DOCS[doc.content_type](doc),
                    }
                ]
            }
            model_input.append(passage)

        embeddings = self._get_predictions(model_input)["passages"]

        return embeddings

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

        dataset, tensor_names, _, __ = self.processor.dataset_from_dicts(
            dicts, indices=list(range(len(dicts))), return_baskets=True
        )
        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        all_embeddings: Dict = {"query": [], "passages": []}

        # FIXME why this was here uncommented? Maybe it goes in the following block
        # self.model.eval()

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

                # Map inputs to their target model
                inputs_by_model = {}
                inputs_by_model["query"] = {
                    name.replace("query_", ""): tensor for name, tensor in batch.items() if name.startswith("query_")
                }

                if "passage_input_ids" in batch.keys():
                    max_seq_len = batch["passage_input_ids"].shape[-1]
                    for content_type in get_args(ContentTypes):
                        content_masks = torch.flatten(batch["content_types"]) == content_type
                        for tensor_name, tensor in batch.items():
                            tensor = tensor.view(-1, max_seq_len)
                            inputs_by_model[content_type][tensor_name.replace("query_", "")] = tensor[content_masks]

                # get logits
                with torch.no_grad():
                    query_embeddings, passage_embeddings = self.model.forward(inputs_by_model=inputs_by_model)[0]
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

    # def train(
    #     self,
    #     data_dir: str,
    #     train_filename: str,
    #     dev_filename: str = None,
    #     test_filename: str = None,
    #     max_samples: int = None,
    #     max_processes: int = 128,
    #     dev_split: float = 0,
    #     batch_size: int = 2,
    #     embed_meta_fields: List[str] = ["page_title", "section_title", "caption"],
    #     num_hard_negatives: int = 1,
    #     num_positives: int = 1,
    #     n_epochs: int = 3,
    #     evaluate_every: int = 1000,
    #     n_gpu: int = 1,
    #     learning_rate: float = 1e-5,
    #     epsilon: float = 1e-08,
    #     weight_decay: float = 0.0,
    #     num_warmup_steps: int = 100,
    #     grad_acc_steps: int = 1,
    #     use_amp: str = None,
    #     optimizer_name: str = "AdamW",
    #     optimizer_correct_bias: bool = True,
    #     save_dir: str = "../saved_models/mm_retrieval",
    #     query_encoder_save_dir: str = "query_encoder",
    #     passage_encoder_save_dir: str = "passage_encoder",
    #     image_encoder_save_dir: str = "image_encoder",
    #     checkpoint_root_dir: Path = Path("model_checkpoints"),
    #     checkpoint_every: Optional[int] = None,
    #     checkpoints_to_keep: int = 3,
    # ):
    #     """
    #     Train a TableTextRetrieval model.
    #     :param data_dir: Directory where training file, dev file and test file are present.
    #     :param train_filename: Training filename.
    #     :param dev_filename: Development set filename, file to be used by model in eval step of training.
    #     :param test_filename: Test set filename, file to be used by model in test step after training.
    #     :param max_samples: Maximum number of input samples to convert. Can be used for debugging a smaller dataset.
    #     :param max_processes: The maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
    #                           It can be set to 1 to disable the use of multiprocessing or make debugging easier.
    #     :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None.
    #     :param batch_size: Total number of samples in 1 batch of data.
    #     :param embed_meta_fields: Concatenate meta fields with each passage and image.
    #                               The default setting in official MMRetrieval embeds page title,
    #                               section title and caption with the corresponding image and title with
    #                               corresponding text passage.
    #     :param num_hard_negatives: Number of hard negative passages (passages which are
    #                                very similar (high score by BM25) to query but do not contain the answer)-
    #     :param num_positives: Number of positive passages.
    #     :param n_epochs: Number of epochs to train the model on.
    #     :param evaluate_every: Number of training steps after evaluation is run.
    #     :param n_gpu: Number of gpus to train on.
    #     :param learning_rate: Learning rate of optimizer.
    #     :param epsilon: Epsilon parameter of optimizer.
    #     :param weight_decay: Weight decay parameter of optimizer.
    #     :param grad_acc_steps: Number of steps to accumulate gradient over before back-propagation is done.
    #     :param use_amp: Whether to use automatic mixed precision (AMP) or not. The options are:
    #                 "O0" (FP32)
    #                 "O1" (Mixed Precision)
    #                 "O2" (Almost FP16)
    #                 "O3" (Pure FP16).
    #                 For more information, refer to: https://nvidia.github.io/apex/amp.html
    #     :param optimizer_name: What optimizer to use (default: TransformersAdamW).
    #     :param num_warmup_steps: Number of warmup steps.
    #     :param optimizer_correct_bias: Whether to correct bias in optimizer.
    #     :param save_dir: Directory where models are saved.
    #     :param query_encoder_save_dir: Directory inside save_dir where query_encoder model files are saved.
    #     :param passage_encoder_save_dir: Directory inside save_dir where passage_encoder model files are saved.
    #     :param image_encoder_save_dir: Directory inside save_dir where image_encoder model files are saved.
    #     """

    #     self.processor.embed_meta_fields = embed_meta_fields
    #     self.processor.data_dir = Path(data_dir)
    #     self.processor.train_filename = train_filename
    #     self.processor.dev_filename = dev_filename
    #     self.processor.test_filename = test_filename
    #     self.processor.max_samples = max_samples
    #     self.processor.dev_split = dev_split
    #     self.processor.num_hard_negatives = num_hard_negatives
    #     self.processor.num_positives = num_positives

    #     if isinstance(self.model, DataParallel):
    #         self.model.module.connect_heads_with_processor(self.processor.tasks, require_labels=True)
    #     else:
    #         self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

    #     data_silo = DataSilo(
    #         processor=self.processor, batch_size=batch_size, distributed=False, max_processes=max_processes
    #     )

    #     # 5. Create an optimizer
    #     self.model, optimizer, lr_schedule = initialize_optimizer(
    #         model=self.model,
    #         learning_rate=learning_rate,
    #         optimizer_opts={
    #             "name": optimizer_name,
    #             "correct_bias": optimizer_correct_bias,
    #             "weight_decay": weight_decay,
    #             "eps": epsilon,
    #         },
    #         schedule_opts={"name": "LinearWarmup", "num_warmup_steps": num_warmup_steps},
    #         n_batches=len(data_silo.loaders["train"]),
    #         n_epochs=n_epochs,
    #         grad_acc_steps=grad_acc_steps,
    #         device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
    #         use_amp=use_amp,
    #     )

    #     # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    #     trainer = Trainer.create_or_load_checkpoint(
    #         model=self.model,
    #         optimizer=optimizer,
    #         data_silo=data_silo,
    #         epochs=n_epochs,
    #         n_gpu=n_gpu,
    #         lr_schedule=lr_schedule,
    #         evaluate_every=evaluate_every,
    #         device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
    #         use_amp=use_amp,
    #         checkpoint_root_dir=Path(checkpoint_root_dir),
    #         checkpoint_every=checkpoint_every,
    #         checkpoints_to_keep=checkpoints_to_keep,
    #     )

    #     # 7. Let it grow! Watch the tracked metrics live on experiment tracker (e.g. Mlflow)
    #     trainer.train()

    #     self.model.save(
    #         Path(save_dir),
    #         lm1_name=query_encoder_save_dir,
    #         lm2_name=passage_encoder_save_dir,
    #         lm3_name=image_encoder_save_dir,
    #     )
    #     self.query_tokenizer.save_pretrained(f"{save_dir}/{query_encoder_save_dir}")
    #     self.passage_tokenizer.save_pretrained(f"{save_dir}/{passage_encoder_save_dir}")
    #     self.image_tokenizer.save_pretrained(f"{save_dir}/{image_encoder_save_dir}")

    #     if len(self.devices) > 1:
    #         self.model = DataParallel(self.model, device_ids=self.devices)

    # def save(
    #     self,
    #     save_dir: Union[Path, str],
    #     query_encoder_dir: str = "query_encoder",
    #     passage_encoder_dir: str = "passage_encoder",
    #     image_encoder_dir: str = "image_encoder",
    # ):
    #     """
    #     Save TableTextRetriever to the specified directory.

    #     :param save_dir: Directory to save to.
    #     :param query_encoder_dir: Directory in save_dir that contains query encoder model.
    #     :param passage_encoder_dir: Directory in save_dir that contains passage encoder model.
    #     :param image_encoder_dir: Directory in save_dir that contains image encoder model.
    #     :return: None
    #     """
    #     save_dir = Path(save_dir)
    #     self.model.save(save_dir, lm1_name=query_encoder_dir, lm2_name=passage_encoder_dir, lm3_name=image_encoder_dir)
    #     save_dir = str(save_dir)
    #     self.query_tokenizer.save_pretrained(save_dir + f"/{query_encoder_dir}")
    #     self.passage_tokenizer.save_pretrained(save_dir + f"/{passage_encoder_dir}")
    #     self.image_tokenizer.save_pretrained(save_dir + f"/{image_encoder_dir}")
