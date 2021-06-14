import logging
from typing import Optional

from haystack.graph_retriever.base import BaseGraphRetriever

from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


class Text2SparqlRetriever(BaseGraphRetriever):
    """
        Graph retriever that uses a pre-trained Bart model to translate natural language questions given in text form to queries in SPARQL format.
        The generated SPARQL query is executed on a knowledge graph.
    """

    def __init__(self, knowledge_graph, model_name_or_path, top_k: int = 1):
        """
        Init the Retriever by providing a knowledge graph and a pre-trained BART model

        :param knowledge_graph: An instance of BaseKnowledgeGraph on which to execute SPARQL queries.
        :param model_name_or_path: Name of or path to a pre-trained BartForConditionalGeneration model.
        :param top_k: How many SPARQL queries to generate per text query.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(knowledge_graph=knowledge_graph, model_name_or_path=model_name_or_path, top_k=top_k)

        self.knowledge_graph = knowledge_graph
        # TODO We should extend this to any seq2seq models and use the AutoModel class
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, forced_bos_token_id=0)
        self.tok = BartTokenizer.from_pretrained(model_name_or_path)
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None):
        """
        Translate a text query to SPARQL and execute it on the knowledge graph to retrieve a list of answers
        
        :param query: Text query that shall be translated to SPARQL and then executed on the knowledge graph
        :param top_k: How many SPARQL queries to generate per text query.
        """
        
        if top_k is None:
            top_k = self.top_k
        inputs = self.tok([query], max_length=100, truncation=True, return_tensors='pt')
        # generate self.top_k+2 SPARQL queries so that we can dismiss some queries with wrong syntax
        temp = self.model.generate(inputs['input_ids'],
                                   num_beams=5,
                                   max_length=100,
                                   num_return_sequences=self.top_k+2,
                                   early_stopping=True)
        sparql_queries = [self.tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in temp]
        answers = []
        for sparql_query in sparql_queries:
            ans, query = self._query_kg(sparql_query=sparql_query)
            if len(ans) > 0:
                answers.append((ans, query))

        # if there are no answers we still want to return something
        if len(answers) == 0:
            answers.append(("", ""))
        results = answers[:self.top_k]
        results = [self.format_result(result) for result in results]
        return results

    def _query_kg(self, sparql_query):
        """
        Execute a single SPARQL query on the knowledge graph to retrieve an answer and unpack different answer styles for boolean queries, count queries, and list queries
        
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
                        for k, v in x.items():
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
