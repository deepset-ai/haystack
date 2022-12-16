import logging
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.base import BaseComponent
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document

logger = logging.getLogger(__name__)


class PseudoLabelGenerator(BaseComponent):
    """
    PseudoLabelGenerator is a component that creates Generative Pseudo Labeling (GPL) training data for the
    training of dense retrievers.

    GPL is an unsupervised domain adaptation method for the training of dense retrievers. It is based on question
    generation and pseudo labelling with powerful cross-encoders. To train a domain-adapted model, it needs access
    to an unlabeled target corpus, usually through DocumentStore and a Retriever to mine for negatives.

    For more details, see [GPL](https://github.com/UKPLab/gpl).

    For example:

    ```python
    document_store = ElasticsearchDocumentStore(...)
    retriever = BM25Retriever(...)
    qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1")
    plg = PseudoLabelGenerator(qg, retriever)
    output, output_id = psg.run(documents=document_store.get_all_documents())
    ```

    Note:

        While the NLP researchers trained the default question
        [generation](https://huggingface.co/doc2query/msmarco-t5-base-v1) and the cross
        [encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) models on
        the English language corpus, we can also use the language-specific question generation and
        cross-encoder models in the target language of our choice to apply GPL to documents in languages
        other than English.

        As of this writing, the German language question
        [generation](https://huggingface.co/ml6team/mt5-small-german-query-generation) and the cross
        [encoder](https://huggingface.co/ml6team/cross-encoder-mmarco-german-distilbert-base) models are
        already available, as well as question [generation](https://huggingface.co/doc2query/msmarco-14langs-mt5-base-v1)
        and the cross [encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
        models trained on fourteen languages.


    """

    outgoing_edges: int = 1

    def __init__(
        self,
        question_producer: Union[QuestionGenerator, List[Dict[str, str]]],
        retriever: BaseRetriever,
        cross_encoder_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_questions_per_document: int = 3,
        top_k: int = 50,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Loads the cross-encoder model and prepares PseudoLabelGenerator.

        :param question_producer: The question producer used to generate questions or a list of already produced
        questions/document pairs in a Dictionary format {"question": "question text ...", "document": "document text ..."}.
        :type question_producer: Union[QuestionGenerator, List[Dict[str, str]]]
        :param retriever: The Retriever used to query document stores.
        :type retriever: BaseRetriever
        :param cross_encoder_model_name_or_path: The path to the cross encoder model, defaults to
        `cross-encoder/ms-marco-MiniLM-L-6-v2`.
        :type cross_encoder_model_name_or_path: str (optional)
        :param max_questions_per_document: The max number of questions generated per document, defaults to 3.
        :type max_questions_per_document: int
        :param top_k: The number of answers retrieved for each question, defaults to 50.
        :type top_k: int (optional)
        :param batch_size: The number of documents to process at a time.
        :type batch_size: int (optional)
        :param progress_bar: Whether to show a progress bar, defaults to True.
        :type progress_bar: bool (optional)
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :type use_auth_token: Union[str, bool] (optional)
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit CrossEncoder inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """

        super().__init__()
        self.question_document_pairs = None
        self.question_generator = None  # type: ignore
        if isinstance(question_producer, QuestionGenerator):
            self.question_generator = question_producer
        elif isinstance(question_producer, list) and len(question_producer) > 0:
            example = question_producer[0]
            if isinstance(example, dict) and "question" in example and "document" in example:
                self.question_document_pairs = question_producer
            else:
                raise ValueError(
                    "The question_producer list must contain dictionaries with keys 'question' and 'document'."
                )
        else:
            raise ValueError("Provide either a QuestionGenerator or a non-empty list of questions/document pairs.")
        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        self.retriever = retriever

        self.cross_encoder = CrossEncoder(
            cross_encoder_model_name_or_path,
            device=str(self.devices[0]),
            tokenizer_args={"use_auth_token": use_auth_token},
            automodel_args={"use_auth_token": use_auth_token},
        )
        self.max_questions_per_document = max_questions_per_document
        self.top_k = top_k
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def generate_questions(self, documents: List[Document], batch_size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        It takes a list of documents and generates a list of question-document pairs.

        :param documents: A list of documents to generate questions from.
        :type documents: List[Document]
        :param batch_size: The number of documents to process at a time.
        :type batch_size: Optional[int]
        :return: A list of question-document pairs.
        """
        question_doc_pairs: List[Dict[str, str]] = []
        if self.question_document_pairs:
            question_doc_pairs = self.question_document_pairs
        else:
            batch_size = batch_size if batch_size else self.batch_size
            questions: List[List[str]] = self.question_generator.generate_batch(  # type: ignore
                [d.content for d in documents], batch_size=batch_size
            )
            for idx, question_list_per_doc in enumerate(questions):
                for q in question_list_per_doc[: self.max_questions_per_document]:  # type: ignore
                    question_doc_pairs.append({"question": q.strip(), "document": documents[idx].content})
        return question_doc_pairs

    def mine_negatives(
        self, question_doc_pairs: List[Dict[str, str]], batch_size: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Given a list of question and positive document pairs, this function returns a list of question/positive document/negative document
        dictionaries.

        :param question_doc_pairs: A list of question/positive document pairs.
        :type question_doc_pairs: List[Dict[str, str]]
        :param batch_size: The number of queries to run in a batch.
        :type batch_size: int (optional)
        :return: A list of dictionaries, where each dictionary contains the question, positive document,
                and negative document.
        """
        question_pos_doc_neg_doc: List[Dict[str, str]] = []
        batch_size = batch_size if batch_size else self.batch_size

        for i in tqdm(
            range(0, len(question_doc_pairs), batch_size), disable=not self.progress_bar, desc="Mine negatives"
        ):
            # question in batches to minimize network latency
            i_end = min(i + batch_size, len(question_doc_pairs))
            queries: List[str] = [e["question"] for e in question_doc_pairs[i:i_end]]
            pos_docs: List[str] = [e["document"] for e in question_doc_pairs[i:i_end]]

            docs: List[List[Document]] = self.retriever.retrieve_batch(
                queries=queries, top_k=self.top_k, batch_size=batch_size
            )

            # iterate through queries and find negatives
            for question, pos_doc, top_docs in zip(queries, pos_docs, docs):
                random.shuffle(top_docs)
                for doc_item in top_docs:
                    neg_doc = doc_item.content
                    if neg_doc != pos_doc:
                        question_pos_doc_neg_doc.append({"question": question, "pos_doc": pos_doc, "neg_doc": neg_doc})
                        break
        return question_pos_doc_neg_doc

    def generate_margin_scores(
        self, mined_negatives: List[Dict[str, str]], batch_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Given a list of mined negatives, this function predicts the score margin between the positive and negative document using
        the cross-encoder.

        The function returns a list of examples, where each example is a dictionary with the following keys:

        * question: The question string.
        * pos_doc: Positive document string (the document containing the answer).
        * neg_doc: Negative document string (the document that doesn't contain the answer).
        * score: The margin between the score for question-positive document pair and the score for question-negative document pair.

        :param mined_negatives: The list of mined negatives.
        :type mined_negatives: List[Dict[str, str]]
        :param batch_size: The number of mined negative lists to run in a batch.
        :type batch_size: int (optional)
        :return: A list of dictionaries, each of which has the following keys:
            - question: The question string
            - pos_doc: Positive document string
            - neg_doc: Negative document string
            - score: The score margin
        """
        examples: List[Dict] = []
        batch_size = batch_size if batch_size else self.batch_size
        for i in tqdm(range(0, len(mined_negatives), batch_size), disable=not self.progress_bar, desc="Score margin"):
            negatives_batch = mined_negatives[i : i + batch_size]
            pb = []
            for item in negatives_batch:
                pb.append([item["question"], item["pos_doc"]])
                pb.append([item["question"], item["neg_doc"]])
            scores = self.cross_encoder.predict(pb)
            for idx, item in enumerate(negatives_batch):
                scores_idx = idx * 2
                score_margin = scores[scores_idx] - scores[scores_idx + 1]
                examples.append(
                    {
                        "question": item["question"],
                        "pos_doc": item["pos_doc"],
                        "neg_doc": item["neg_doc"],
                        "score": score_margin,
                    }
                )
        return examples

    def generate_pseudo_labels(self, documents: List[Document], batch_size: Optional[int] = None) -> Tuple[dict, str]:
        """
        Given a list of documents, this function generates a list of question-document pairs, mines for negatives, and
        scores a positive/negative margin with cross-encoder. The output is the training data for the
        adaptation of dense retriever models.

        :param documents: List[Document] = The list of documents to mine negatives from.
        :type documents: List[Document]
        :param batch_size: The number of documents to process in a batch.
        :type batch_size: Optional[int]
        :return: A dictionary with a single key 'gpl_labels' representing a list of dictionaries, where each
        dictionary contains the following keys:
            - question: The question string.
            - pos_doc: Positive document for the given question.
            - neg_doc: Negative document for the given question.
            - score: The margin between the score for question-positive document pair and the score for question-negative document pair.
        """
        # see https://github.com/UKPLab/gpl for more information about GPL algorithm
        batch_size = batch_size if batch_size else self.batch_size

        # step 1: generate questions
        question_doc_pairs = self.generate_questions(documents=documents, batch_size=batch_size)

        # step 2: negative mining
        mined_negatives = self.mine_negatives(question_doc_pairs=question_doc_pairs, batch_size=batch_size)

        # step 3: pseudo labeling (scoring) with cross-encoder
        pseudo_labels: List[Dict[str, str]] = self.generate_margin_scores(mined_negatives, batch_size=batch_size)
        return {"gpl_labels": pseudo_labels}, "output_1"

    def run(self, documents: List[Document]) -> Tuple[dict, str]:  # type: ignore
        return self.generate_pseudo_labels(documents=documents)

    def run_batch(self, documents: Union[List[Document], List[List[Document]]]) -> Tuple[dict, str]:  # type: ignore
        flat_list_of_documents = []
        for sub_list_documents in documents:
            if isinstance(sub_list_documents, Iterable):
                flat_list_of_documents += sub_list_documents
            else:
                flat_list_of_documents.append(sub_list_documents)
        return self.generate_pseudo_labels(documents=flat_list_of_documents)
