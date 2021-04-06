import logging
from typing import Optional

from haystack.graph_retriever.base import BaseGraphRetriever

from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


class Text2SparqlRetriever(BaseGraphRetriever):
    def __init__(self, knowledge_graph, model_name_or_path, top_k: int = 1):
        self.knowledge_graph = knowledge_graph
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, force_bos_token_to_be_generated=True)
        self.tok = BartTokenizer.from_pretrained(model_name_or_path)
        self.top_k = top_k

    def retrieve(self, question_text: str, top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        inputs = self.tok([question_text], max_length=100, truncation=True, return_tensors='pt')
        temp = self.model.generate(inputs['input_ids'],
                                   num_beams=self.top_k,
                                   max_length=100,
                                   num_return_sequences=self.top_k,
                                   early_stopping=True)
        sparql_list = [self.tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in temp]
        answers = []
        for s in sparql_list:
            ans,query = self._query_kg(query=s)
            if len(ans) > 0:
                answers.append((ans,query))

        # if there are no answers we still want to return something
        if len(answers) == 0:
            answers.append(("",""))
        results = answers[:self.top_k]
        results = [self.format_result(result) for result in results]
        return results

    def run(self, query, **kwargs):
        answers = self.retrieve(question_text=query)

        results = {"query": query,
                   "answers": answers,
                   **kwargs}
        return results, "output_1"


    def _query_kg(self, query):
        # query KG
        try:
            response = self.knowledge_graph.query(query=query)

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

        return result, query

    def format_result(self, result):
        """
        Generate formatted dictionary output with text answer and additional info
        """
        query = result[1]
        prediction = result[0]
        meta = {"model": self.__class__.__name__}
        meta["sparql_query"] = query

        return {"answer": prediction, "meta": meta}
