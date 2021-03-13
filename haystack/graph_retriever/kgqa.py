import logging
import re

import pandas as pd

from haystack.graph_retriever.base import BaseGraphRetriever
from haystack.graph_retriever.query_executor import QueryExecutor

from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


class Text2SparqlRetriever(BaseGraphRetriever):
    def __init__(self, knowledge_graph, model_name_or_path, top_k: int = 1):
        self.knowledge_graph = knowledge_graph
        self.query_executor: QueryExecutor = QueryExecutor(knowledge_graph)
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, force_bos_token_to_be_generated=True)
        self.tok = BartTokenizer.from_pretrained(model_name_or_path)
        self.top_k = top_k

    def retrieve(self, question_text: str):
        logger.info("logging info...")
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
        # Bring generated query into Harry Potter KG form
        query = query.replace("}", " }")
        query = query.replace(")", " )")
        splits = query.split()
        query = ""
        for s in splits:
            if s.startswith("hp:"):
                s = s.replace("hp:", "<https://deepset.ai/harry_potter/")
                s = s + ">"
            query += s + " "

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
        text_answer = str(self.prediction_to_text(prediction))
        meta = {"model": self.__class__.__name__}
        meta["urls"] = str(self.prediction_to_urls(prediction))
        meta["sparql_query"] = query
        if len(str(text_answer).split("\n")) > 4:
            meta["long_answer_list"] = True
        else:
            meta["long_answer_list"] = False

        return {"answer": text_answer, "meta": meta}

    def eval(self, filename, question_type, top_k_graph):
        """
        Calculate top_k accuracy given a tsv file with question and answer columns and store predictions
        Do this evaluation for one chosen type of questions (List, Boolean, Count)
        """
        df = pd.read_csv(filename)
        df = df[df['Type'] == question_type]
        predictions_for_all_queries = []
        number_of_correct_predictions = 0
        correct_predictions = []
        for index, row in df.iterrows():
            predictions_for_query = self.retrieve(question_text=row['Question'], top_k=top_k_graph)
            if not predictions_for_query:
                predictions_for_all_queries.append(None)
                correct_predictions.append(0)
                continue
            top_k_contain_correct_answer = False
            for prediction in predictions_for_query:
                top_k_contain_correct_answer = self.compare_answers(row['Answer'], prediction, row['Type'])
                if top_k_contain_correct_answer:
                    number_of_correct_predictions += 1
                    correct_predictions.append(1)
                    logger.info("Correct answer.")
                    break
            if not top_k_contain_correct_answer:
                logger.info(f"Wrong answer(s). Expected {row['Answer']}")
                correct_predictions.append(0)
            predictions_for_all_queries.append(predictions_for_query)

        logger.info(f"{number_of_correct_predictions} correct answers out of {len(df)} for k={top_k_graph}")
        df['prediction'] = predictions_for_all_queries
        df['correct'] = correct_predictions
        df.to_csv("predictions.csv", index=False)

    def compare_answers(self, answer, prediction, question_type: str):
        if question_type == "List":
            if isinstance(prediction, int):
                # assumed wrong question type
                return False
            # split multiple entities
            # strip whitespaces, lowercase, and remove namespace
            # convert answers to sets so that the order of the items does not matter
            answer = [str(answer_item).strip().lower() for answer_item in re.split(",|\ \ |\n", str(answer))]
            answer = {answer_item.replace('https://harrypotter.fandom.com/wiki/', "").replace("_", " ").replace("-", " ") for answer_item in answer}

            prediction = [str(prediction_item).strip().lower() for prediction_item in prediction]
            prediction = {prediction_item.replace('https://deepset.ai/harry_potter/', "").replace("_", " ") for prediction_item in prediction}

        elif question_type == "Boolean":
            answer = bool(answer)
            prediction = bool(prediction)
        elif question_type == "Count":
            answer = int(answer)
            prediction = int(prediction)

        return answer == prediction