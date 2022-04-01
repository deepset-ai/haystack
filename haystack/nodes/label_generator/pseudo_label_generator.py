import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm
from haystack.nodes.base import BaseComponent
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document


class PseudoLabelGenerator(BaseComponent):
    """
    The PseudoLabelGenerator is a component that creates Generative Pseudo Labeling (GPL) training data for the
    training of dense retrievers.

    GPL is an unsupervised domain adaptation method for the training of dense retrievers. It is based on question
    generation and pseudo labelling with powerful cross-encoders. To train a domain-adapted model, it needs access
    to an unlabeled target corpus, usually via DocumentStore and a retriever to mine for negatives.

    For more details see [https://github.com/UKPLab/gpl](https://github.com/UKPLab/gpl)

    For example:

    ```python
    |   document_store = DocumentStore(...)
    |   retriever = Retriever(...)
    |   qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1")
    |   plg = PseudoLabelGenerator(qg, retriever)
    |   output, output_id = psg.run(documents=document_store.get_all_documents())
    |
    ```
    """

    def __init__(
        self,
        question_producer: Union[QuestionGenerator, List[Dict[str, str]]],
        retriever: BaseRetriever,
        cross_encoder_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_questions_per_document: int = 3,
        top_k: int = 50,
        batch_size: int = 4,
        progress_bar: bool = True,
    ):
        """
        Loads the cross encoder model and prepares PseudoLabelGenerator.

        :param question_producer: The question producer used to generate questions or a list of already produced
        questions/document pairs in Dict format {"question": "question text ...", "document": "document text ..."}.
        :type question_producer: Union[QuestionGenerator, List[Dict[str, str]]]
        :param retriever: The retriever used to query document stores
        :type retriever: BaseRetriever
        :param cross_encoder_model_name_or_path: The path to the cross encoder model, defaults to
        cross-encoder/ms-marco-MiniLM-L-6-v2
        :type cross_encoder_model_name_or_path: str (optional)
        :param max_questions_per_document: The max number of questions generated per document, defaults to 3
        :type max_questions_per_document: int
        :param top_k: The number of answers retrieved for each question, defaults to 50
        :type top_k: int (optional)
        :param batch_size: Number of documents to process at a time
        :type batch_size: int (optional)
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
                raise ValueError("question_producer list must contain dicts with keys 'question' and 'document'")
        else:
            raise ValueError("Provide either a QuestionGenerator or nonempty list of questions/document pairs")

        self.retriever = retriever
        self.cross_encoder = CrossEncoder(cross_encoder_model_name_or_path)
        self.max_questions_per_document = max_questions_per_document
        self.top_k = top_k
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def generate_questions(self, documents: List[Document], batch_size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        It takes a list of documents and generates a list of question-document pairs.

        :param documents: A list of documents to generate questions from
        :type documents: List[Document]
        :param batch_size: Number of documents to process at a time.
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
        Given a list of question and pos_doc pairs, this function returns a list of question/pos_doc/neg_doc
        dictionaries.

        :param question_doc_pairs: A list of question/pos_doc pairs
        :type question_doc_pairs: List[Dict[str, str]]
        :param batch_size: The number of queries to run in a batch
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
        Given a list of mined negatives, predict the score margin between the positive and negative document using
        the cross encoder.

        The function returns a list of examples, where each example is a dictionary with the following keys:

        * question: the question string
        * pos_doc: the positive document string
        * neg_doc: the negative document string
        * score: the score margin

        :param mined_negatives: List of mined negatives
        :type mined_negatives: List[Dict[str, str]]
        :param batch_size: The number of mined negative lists to run in a batch
        :type batch_size: int (optional)
        :return: A list of dictionaries, each of which has the following keys:
            - question: The question string
            - pos_doc: The positive document string
            - neg_doc: The negative document string
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
        Given a list of documents, generate a list of question-document pairs, mine for negatives, and
        score positive/negative margin with cross-encoder. The output is the training data for the
        adaptation of dense retriever models.

        :param documents: List[Document] = List of documents to mine negatives from
        :type documents: List[Document]
        :param batch_size: The number of documents to process in a batch
        :type batch_size: Optional[int]
        :return: A dictionary with a single key 'gpl_labels' representing a list of dictionaries, where each
        dictionary contains the following keys:
            - question: the question
            - pos_doc: the positive document for the given question
            - neg_doc: the negative document for the given question
            - score: the margin score (a float)
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
