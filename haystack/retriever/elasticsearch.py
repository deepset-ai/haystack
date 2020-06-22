import logging
from typing import List, Union

from farm.infer import Inferencer

from haystack.database.base import Document
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store: ElasticsearchDocumentStore, custom_query: str = None):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param custom_query: query string as per Elasticsearch DSL with a mandatory question placeholder($question).

                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().

                             An example custom_query:
                            {
                                "size": 10,
                                "query": {
                                    "bool": {
                                        "should": [{"multi_match": {
                                            "query": "${question}",                 // mandatory $question placeholder
                                            "type": "most_fields",
                                            "fields": ["text", "title"]}}],
                                        "filter": [                                 // optional custom filters
                                            {"terms": {"year": "${years}"}},
                                            {"terms": {"quarter": "${quarters}"}}],
                                    }
                                },
                            }

                             For this custom_query, a sample retrieve() could be:
                             self.retrieve(query="Why did the revenue increase?",
                                           filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
        """
        self.document_store = document_store  # type: ignore
        self.custom_query = custom_query

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            index = self.document_store.index

        documents = self.document_store.query(query, filters, top_k, self.custom_query, index)
        logger.info(f"Got {len(documents)} candidates from retriever")

        return documents

    def eval(
        self,
        label_index: str = "feedback",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k: int = 10,
    ) -> dict:
        """
        Performs evaluation on the Retriever.
        Retriever is evaluated based on whether it finds the correct document given the question string and at which
        position in the ranking of documents the correct document is.

        Returns a dict containing the following metrics:
            - "recall": Proportion of questions for which correct document is among retrieved documents
            - "mean avg precision": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.

        :param label_index: Index/Table in DocumentStore where labeled questions are stored
        :param doc_index: Index/Table in DocumentStore where documents that are used for evaluation are stored
        :param top_k: How many documents to return per question
        """

        # extract all questions for evaluation
        filter = {"origin": label_origin}
        questions = self.document_store.get_all_documents_in_index(index=label_index, filters=filter)

        # calculate recall and mean-average-precision
        correct_retrievals = 0
        summed_avg_precision = 0
        for q_idx, question in enumerate(questions):
            question_string = question["_source"]["question"]
            retrieved_docs = self.retrieve(question_string, top_k=top_k, index=doc_index)
            # check if correct doc in retrieved docs
            for doc_idx, doc in enumerate(retrieved_docs):
                if doc.meta["doc_id"] == question["_source"]["doc_id"]:
                    correct_retrievals += 1
                    summed_avg_precision += 1 / (doc_idx + 1)  # type: ignore
                    break

        number_of_questions = q_idx + 1
        recall = correct_retrievals / number_of_questions
        mean_avg_precision = summed_avg_precision / number_of_questions

        logger.info((f"For {correct_retrievals} out of {number_of_questions} questions ({recall:.2%}), the answer was in"
                     f" the top-{top_k} candidate passages selected by the retriever."))

        return {"recall": recall, "map": mean_avg_precision}


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: ElasticsearchDocumentStore,
        embedding_model: str,
        gpu: bool = True,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
    ):
        """
        TODO
        :param document_store:
        :param embedding_model:
        :param gpu:
        :param model_format:
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
                extraction_layer=self.emb_extraction_layer, gpu=gpu, batch_size=4, max_seq_len=512, num_processes=0
            )

        elif model_format == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            if gpu:
                device = "cuda"
            else:
                device = "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        else:
            raise NotImplementedError

    def retrieve(self, query: str, candidate_doc_ids: List[str] = None, top_k: int = 10) -> List[Document]:  # type: ignore
        query_emb = self.create_embedding(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb[0], top_k, candidate_doc_ids)

        return documents

    def create_embedding(self, texts: Union[List[str], str]) -> List[List[float]]:
        """
        Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)
        :param texts: texts to embed
        :return: list of embeddings (one per input text). Each embedding is a list of floats.
        """

        # for backward compatibility: cast pure str input
        if type(texts) == str:
            texts = [texts]  # type: ignore
        assert type(texts) == list, "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        if self.model_format == "farm" or self.model_format == "transformers":
            res = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])  # type: ignore
            emb = [list(r["vec"]) for r in res] #cast from numpy
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            # get back list of numpy embedding vectors
            res = self.embedding_model.encode(texts)  # type: ignore
            emb = [list(r.astype('float64')) for r in res] #cast from numpy
        return emb
