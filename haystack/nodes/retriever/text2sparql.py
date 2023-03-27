from typing import Optional, List, Union

import logging
from transformers import BartForConditionalGeneration, BartTokenizer

from haystack.document_stores import BaseKnowledgeGraph
from haystack.nodes.retriever.base import BaseGraphRetriever


logger = logging.getLogger(__name__)


class Text2SparqlRetriever(BaseGraphRetriever):
    """
    Graph retriever that uses a pre-trained Bart model to translate natural language questions
    given in text form to queries in SPARQL format.
    The generated SPARQL query is executed on a knowledge graph.
    """

    def __init__(
        self,
        knowledge_graph: BaseKnowledgeGraph,
        model_name_or_path: Optional[str] = None,
        model_version: Optional[str] = None,
        top_k: int = 1,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Init the Retriever by providing a knowledge graph and a pre-trained BART model

        :param knowledge_graph: An instance of BaseKnowledgeGraph on which to execute SPARQL queries.
        :param model_name_or_path: Name of or path to a pre-trained BartForConditionalGeneration model.
        :param model_version: The version of the model to use for entity extraction.
        :param top_k: How many SPARQL queries to generate per text query.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        self.knowledge_graph = knowledge_graph
        # TODO We should extend this to any seq2seq models and use the AutoModel class
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name_or_path, forced_bos_token_id=0, use_auth_token=use_auth_token, revision=model_version
        )
        self.tok = BartTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None):
        """
        Translate a text query to SPARQL and execute it on the knowledge graph to retrieve a list of answers

        :param query: Text query that shall be translated to SPARQL and then executed on the knowledge graph
        :param top_k: How many SPARQL queries to generate per text query.
        """
        if top_k is None:
            top_k = self.top_k
        inputs = self.tok([query], max_length=100, truncation=True, return_tensors="pt")
        # generate top_k+2 SPARQL queries so that we can dismiss some queries with wrong syntax
        temp = self.model.generate(
            inputs["input_ids"], num_beams=5, max_length=100, num_return_sequences=top_k + 2, early_stopping=True
        )
        sparql_queries = [
            self.tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in temp
        ]
        answers = []
        for sparql_query in sparql_queries:
            ans, query = self._query_kg(sparql_query=sparql_query)
            if len(ans) > 0:
                answers.append((ans, query))

        # if there are no answers we still want to return something
        if len(answers) == 0:
            answers.append(("", ""))
        results = answers[:top_k]
        results = [self.format_result(result) for result in results]
        return results

    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None):
        """
        Translate a list of queries to SPARQL and execute it on the knowledge graph to retrieve
        a list of lists of answers (one per query).

        :param queries: List of queries that shall be translated to SPARQL and then executed on the
                        knowledge graph.
        :param top_k: How many SPARQL queries to generate per text query.
        """
        # TODO: This method currently just calls the retrieve method multiple times, so there is room for improvement.

        results = []
        for query in queries:
            cur_result = self.run(query=query, top_k=top_k)
            results.append(cur_result)

        return results

    def _query_kg(self, sparql_query):
        """
        Execute a single SPARQL query on the knowledge graph to retrieve an answer and unpack
        different answer styles for boolean queries, count queries, and list queries.

        :param sparql_query: SPARQL query that shall be executed on the knowledge graph
        """
        try:
            response = self.knowledge_graph.query(sparql_query=sparql_query)

            # unpack different answer styles
            if isinstance(response, list):
                if len(response) == 0:
                    result = ""
                else:
                    result = []
                    for x in response:
                        for v in x.values():
                            result.append(v["value"])
            elif isinstance(response, bool):
                result = str(response)
            elif "count" in response[0]:
                result = str(int(response[0]["count"]["value"]))
            else:
                result = ""

        except Exception:
            result = ""

        return result, sparql_query

    def format_result(self, result):
        """
        Generate formatted dictionary output with text answer and additional info

        :param result: The result of a SPARQL query as retrieved from the knowledge graph
        """
        query = result[1]
        prediction = result[0]
        prediction_meta = {"model": self.__class__.__name__, "sparql_query": query}

        return {"answer": prediction, "prediction_meta": prediction_meta}

    def eval(self):
        raise NotImplementedError
