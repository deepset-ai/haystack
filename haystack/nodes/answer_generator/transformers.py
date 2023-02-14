from typing import Dict, List, Optional, Union

import logging
from collections.abc import Callable
import numpy
import torch
from transformers import (
    RagTokenizer,
    RagTokenForGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    BatchEncoding,
)

from haystack.schema import Document
from haystack.nodes.answer_generator.base import BaseGenerator
from haystack.nodes.retriever.dense import DensePassageRetriever
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


class RAGenerator(BaseGenerator):
    """
    Implementation of Facebook's Retrieval-Augmented Generator (https://arxiv.org/abs/2005.11401) based on
    HuggingFace's transformers (https://huggingface.co/transformers/model_doc/rag.html).

    Instead of "finding" the answer within a document, these models **generate** the answer.
    In that sense, RAG follows a similar approach as GPT-3 but it comes with two huge advantages
    for real-world applications:
    a) it has a manageable model size
    b) the answer generation is conditioned on retrieved documents,
    i.e. the model can easily adjust to domain documents even after training has finished
    (in contrast: GPT-3 relies on the web data seen during training)

    **Example**

     ```python
     query = "who got the first nobel prize in physics?"

     # Retrieve related documents from retriever
     retrieved_docs = retriever.retrieve(query=query)

     # Now generate answer from query and retrieved documents
     generator.predict(
        query=query,
        documents=retrieved_docs,
        top_k=1
     )

     # Answer

     {'query': 'who got the first nobel prize in physics',
      'answers':
          [{'query': 'who got the first nobel prize in physics',
            'answer': ' albert einstein',
            'meta': { 'doc_ids': [...],
                      'doc_scores': [80.42758 ...],
                      'doc_probabilities': [40.71379089355469, ...
                      'content': ['Albert Einstein was a ...]
                      'titles': ['"Albert Einstein"', ...]
      }}]}
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/rag-token-nq",
        model_version: Optional[str] = None,
        retriever: Optional[DensePassageRetriever] = None,
        generator_type: str = "token",
        top_k: int = 2,
        max_length: int = 200,
        min_length: int = 2,
        num_beams: int = 2,
        embed_title: bool = True,
        prefix: Optional[str] = None,
        use_gpu: bool = True,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Load a RAG model from Transformers along with passage_embedding_model.
        See https://huggingface.co/transformers/model_doc/rag.html for more details

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
                                   'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
                                   See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param retriever: `DensePassageRetriever` used to embedded passages for the docs passed to `predict()`. This is optional and is only needed if the docs you pass don't already contain embeddings in `Document.embedding`.
        :param generator_type: Which RAG generator implementation to use ("token" or "sequence")
        :param top_k: Number of independently generated text to return
        :param max_length: Maximum length of generated text
        :param min_length: Minimum length of generated text
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param embed_title: Embedded the title of passage while generating embedding
        :param prefix: The prefix used by the generator's tokenizer.
        :param use_gpu: Whether to use GPU. Falls back on CPU if no GPU is available.
        :param progress_bar: Whether to show a tqdm progress bar or not.
        :param use_auth_token:  The API token used to download private models from Huggingface.
                                If this parameter is set to `True`, then the token generated when running
                                `transformers-cli login` (stored in ~/.huggingface) will be used.
                                Additional information can be found here
                                https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(progress_bar=progress_bar)

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.min_length = min_length
        self.generator_type = generator_type
        self.num_beams = num_beams
        self.embed_title = embed_title
        self.prefix = prefix
        self.retriever = retriever

        if top_k > self.num_beams:
            top_k = self.num_beams
            logger.warning("top_k value should not be greater than num_beams, hence setting it to %s", num_beams)

        self.top_k = top_k

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        self.tokenizer = RagTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)

        if self.generator_type == "sequence":
            raise NotImplementedError("RagSequenceForGeneration is not implemented yet")
            # TODO: Enable when transformers have it. Refer https://github.com/huggingface/transformers/issues/7905
            # Also refer refer https://github.com/huggingface/transformers/issues/7829
            # self.model = RagSequenceForGeneration.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)

        self.model = RagTokenForGeneration.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        self.model.to(str(self.devices[0]))

    # Copied cat_input_and_doc method from transformers.RagRetriever
    # Refer section 2.3 of https://arxiv.org/abs/2005.11401
    def _cat_input_and_doc(self, doc_title: str, doc_text: str, input_string: str, prefix: Optional[str]):
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        if prefix is None:
            prefix = ""
        out = (
            prefix + doc_title + self.model.config.title_sep + doc_text + self.model.config.doc_sep + input_string
        ).replace("  ", " ")

        return out

    # Copied postprocess_docs method from transformers.RagRetriever and modified
    def _get_contextualized_inputs(
        self, texts: List[str], query: str, titles: Optional[List[str]] = None, return_tensors: str = "pt"
    ):
        titles_list = titles if self.embed_title and titles is not None else [""] * len(texts)
        prefix = self.prefix if self.prefix is not None else self.model.config.generator.prefix

        rag_input_strings = [
            self._cat_input_and_doc(doc_title=titles_list[i], doc_text=texts[i], input_string=query, prefix=prefix)
            for i in range(len(texts))
        ]

        contextualized_inputs = self.tokenizer.generator(
            rag_input_strings,
            max_length=self.model.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"].to(self.devices[0]), contextualized_inputs["attention_mask"].to(
            self.devices[0]
        )

    def _prepare_passage_embeddings(self, docs: List[Document], embeddings: numpy.ndarray) -> torch.Tensor:
        # If document missing embedding, then need embedding for all the documents
        is_embedding_required = embeddings is None or any(embedding is None for embedding in embeddings)

        if is_embedding_required:
            if self.retriever is None:
                raise AttributeError(
                    "_prepare_passage_embeddings need a DPR instance as self.retriever to embed document"
                )

            embeddings = self.retriever.embed_documents(docs)

        embeddings_in_tensor = torch.cat(
            [torch.from_numpy(embedding).float().unsqueeze(0) for embedding in embeddings], dim=0
        )

        return embeddings_in_tensor.to(self.devices[0])

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Generate the answer to the input query. The generation will be conditioned on the supplied documents.
        These documents can for example be retrieved via the Retriever.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers plus additional infos in a dict like this:

        ```python
        {'query': 'who got the first nobel prize in physics',
         'answers':
             [{'query': 'who got the first nobel prize in physics',
               'answer': ' albert einstein',
               'meta': { 'doc_ids': [...],
                         'doc_scores': [80.42758 ...],
                         'doc_probabilities': [40.71379089355469, ...
                         'content': ['Albert Einstein was a ...]
                         'titles': ['"Albert Einstein"', ...]
         }}]}
        ```
        """
        torch.set_grad_enabled(False)
        if len(documents) == 0:
            raise AttributeError("generator need documents to predict the answer")

        top_k = top_k if top_k is not None else self.top_k

        if top_k > self.num_beams:
            top_k = self.num_beams
            logger.warning("top_k value should not be greater than num_beams, hence setting it to %s", top_k)

        # Flatten the documents so easy to reference
        flat_docs_dict = self._flatten_docs(documents)

        # Extract title
        titles = [d.get("name", "") for d in flat_docs_dict["meta"]]

        # Raw document embedding and set device of query_embedding
        passage_embeddings = self._prepare_passage_embeddings(docs=documents, embeddings=flat_docs_dict["embedding"])

        # Query tokenization
        input_dict = self.tokenizer(text=[query], return_tensors="pt", padding="longest", truncation=True)

        input_ids = input_dict["input_ids"].to(self.devices[0])
        # Query embedding
        query_embedding = self.model.question_encoder(input_ids)[0]

        # Prepare contextualized input_ids of documents
        # (will be transformed into contextualized inputs inside generator)
        context_input_ids, context_attention_mask = self._get_contextualized_inputs(
            texts=flat_docs_dict["content"], titles=titles, query=query
        )

        # Compute doc scores from docs_embedding
        doc_scores = torch.bmm(query_embedding.unsqueeze(1), passage_embeddings.unsqueeze(0).transpose(1, 2)).squeeze(1)

        # Get generated ids from generator
        generator_ids = self.model.generate(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_return_sequences=top_k,
            num_beams=self.num_beams,
            max_length=self.max_length,
            min_length=self.min_length,
            n_docs=len(flat_docs_dict["content"]),
        )

        generated_answers = self.tokenizer.batch_decode(generator_ids, skip_special_tokens=True)
        answers = self._create_answers(generated_answers, documents)
        result = {"query": query, "answers": answers}

        return result


class Seq2SeqGenerator(BaseGenerator):

    """
    A generic sequence-to-sequence generator based on HuggingFace's transformers.

    This generator supports all [Text2Text](https://huggingface.co/models?pipeline_tag=text2text-generation) models
    from the Hugging Face hub. If the primary interface for the model specified by `model_name_or_path` constructor
    parameter is AutoModelForSeq2SeqLM from Hugging Face, then you can use it in this Generator.

    Moreover, as language models prepare model input in their specific encoding, each model
    specified with model_name_or_path parameter in this Seq2SeqGenerator should have an
    accompanying model input converter that takes care of prefixes, separator tokens etc.
    By default, we provide model input converters for a few well-known seq2seq language models (e.g. ELI5).
    It is the responsibility of Seq2SeqGenerator user to ensure an appropriate model input converter
    is either already registered or specified on a per-model basis in the Seq2SeqGenerator constructor.

    For mode details on custom model input converters refer to _BartEli5Converter

    For a list of all text2text-generation models, see
    the [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text2text-generation)


    **Example**

     ```python
     query = "Why is Dothraki language important?"

     # Retrieve related documents from retriever
     retrieved_docs = retriever.retrieve(query=query)

     # Now generate answer from query and retrieved documents
     generator.predict(
        query=query,
        documents=retrieved_docs,
        top_k=1
     )

     # Answer

     {'query': 'who got the first nobel prize in physics',
      'answers':
          [{'query': 'who got the first nobel prize in physics',
            'answer': ' albert einstein',
            'meta': { 'doc_ids': [...],
                      'doc_scores': [80.42758 ...],
                      'doc_probabilities': [40.71379089355469, ...
                      'content': ['Albert Einstein was a ...]
                      'titles': ['"Albert Einstein"', ...]
      }}]}
    ```
    """

    _model_input_converters: Dict[str, Callable] = {}

    def __init__(
        self,
        model_name_or_path: str,
        input_converter: Optional[Callable] = None,
        top_k: int = 1,
        max_length: int = 200,
        min_length: int = 2,
        num_beams: int = 8,
        use_gpu: bool = True,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        :param model_name_or_path: a HF model name for auto-regressive language model like GPT2, XLNet, XLM, Bart, T5 etc
        :param input_converter: an optional Callable to prepare model input for the underlying language model
                                specified in model_name_or_path parameter. The required __call__ method signature for
                                the Callable is:
                                __call__(tokenizer: PreTrainedTokenizer, query: str, documents: List[Document],
                                top_k: Optional[int] = None) -> BatchEncoding:
        :param top_k: Number of independently generated text to return
        :param max_length: Maximum length of generated text
        :param min_length: Minimum length of generated text
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param use_gpu: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
        :param progress_bar: Whether to show a tqdm progress bar or not.
        :param use_auth_token:  The API token used to download private models from Huggingface.
                                If this parameter is set to `True`, then the token generated when running
                                `transformers-cli login` (stored in ~/.huggingface) will be used.
                                Additional information can be found here
                                https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(progress_bar=progress_bar)
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams

        if top_k > self.num_beams:
            top_k = self.num_beams
            logger.warning("top_k value should not be greater than num_beams, hence setting it to %s", num_beams)

        self.top_k = top_k

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        Seq2SeqGenerator._register_converters(model_name_or_path, input_converter)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        self.model.to(str(self.devices[0]))
        self.model.eval()

    @classmethod
    def _register_converters(cls, model_name_or_path: str, custom_converter: Optional[Callable]):
        # init if empty
        if not cls._model_input_converters:
            for c in ["yjernite/bart_eli5", "vblagoje/bart_lfqa"]:
                cls._model_input_converters[c] = _BartEli5Converter()

        # register user provided custom converter
        if model_name_or_path and custom_converter:
            cls._model_input_converters[model_name_or_path] = custom_converter

    @classmethod
    def _get_converter(cls, model_name_or_path: str) -> Optional[Callable]:
        return cls._model_input_converters.get(model_name_or_path)

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Generate the answer to the input query. The generation will be conditioned on the supplied documents.
        These document can be retrieved via the Retriever or supplied directly via predict method.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers

        """
        torch.set_grad_enabled(False)
        if len(documents) == 0:
            raise AttributeError("generator needs documents to predict the answer")

        top_k = top_k if top_k is not None else self.top_k

        if top_k > self.num_beams:
            top_k = self.num_beams
            logger.warning("top_k value should not be greater than num_beams, hence setting it to %s", top_k)

        converter: Optional[Callable] = Seq2SeqGenerator._get_converter(self.model_name_or_path)
        if converter is None:
            raise KeyError(
                f"Seq2SeqGenerator doesn't have input converter registered for {self.model_name_or_path}. "
                f"Provide custom converter for {self.model_name_or_path} in Seq2SeqGenerator initialization"
            )

        try:
            query_and_docs_encoded: BatchEncoding = converter(
                tokenizer=self.tokenizer, query=query, documents=documents, top_k=top_k
            ).to(self.devices[0])
        except TypeError:
            raise TypeError(
                f"Language model input converter {converter} provided in Seq2SeqGenerator.__init__() does "
                f"not have a valid __call__ method signature. The required Callable __call__ signature is: "
                f"__call__(tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], "
                f"top_k: Optional[int] = None) -> BatchEncoding:"
            )

        generated_answers_encoded = self.model.generate(
            input_ids=query_and_docs_encoded["input_ids"],
            attention_mask=query_and_docs_encoded["attention_mask"],
            min_length=self.min_length,
            max_length=self.max_length,
            do_sample=True if self.num_beams == 1 else False,
            early_stopping=True,
            num_beams=self.num_beams,
            temperature=1.0,
            top_k=None,
            top_p=None,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=top_k,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

        generated_answers = self.tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True)
        answers = self._create_answers(generated_answers, documents)
        result = {"query": query, "answers": answers}

        return result


class _BartEli5Converter:
    """
    A sequence-to-sequence model input converter (https://huggingface.co/yjernite/bart_eli5) based on the
    BART architecture fine-tuned on ELI5 dataset (https://arxiv.org/abs/1907.09190).

    The converter takes documents and a query as input and formats them into a single sequence
    that a seq2seq model can use it as input for its generation step.
    This includes model-specific prefixes, separation tokens and the actual conversion into tensors.

    For more details refer to Yacine Jernite's excellent LFQA contributions at https://yjernite.github.io/lfqa.html
    """

    def __call__(
        self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        conditioned_doc = "<P> " + " <P> ".join([d.content for d in documents])

        # concatenate question and support document into BART input
        query_and_docs = "question: {} context: {}".format(query, conditioned_doc)

        return tokenizer([(query_and_docs, "A")], truncation=True, padding=True, return_tensors="pt")
