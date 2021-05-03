import json
import pandas as pd
from tqdm import tqdm

COLUMN_NAMES = ["title", "context", "question", "id", "answer_text", "answer_start", "is_impossible"]

class SquadData:
    def __init__(self, squad_data):
        if type(squad_data) == dict:
            self.version = squad_data.get("version")
            self.data = squad_data["data"]
        elif type(squad_data) == list:
            self.version = None
            self.data = squad_data

    @classmethod
    def from_file(cls, filename):
        data = json.load(open(filename))
        return cls(data)

    def slice_data(self, start=None, end=None):
        return self.data[start:end]

    def save(self, filename):
        with open(filename, "w") as f:
            squad_data = {"version": self.version, "data": self.data}
            json.dump(squad_data, f, indent=2)

    def to_df(self):
        flat = []
        for document in self.data:
            title = document["title"]
            for paragraph in document["paragraphs"]:
                context = paragraph["context"]
                for question in paragraph["qas"]:
                    q = question["question"]
                    id = question["id"]
                    is_impossible = question["is_impossible"]
                    # For no_answer samples
                    if len(question["answers"]) == 0:
                        flat.append({"title": title,
                                     "context": context,
                                     "question": q,
                                     "id": id,
                                     "answer_text": "",
                                     "answer_start": None,
                                     "is_impossible": is_impossible})
                    # For span answer samples
                    else:
                        for answer in question["answers"]:
                            answer_text = answer["text"]
                            answer_start = answer["answer_start"]
                            flat.append({"title": title,
                                         "context": context,
                                         "question": q,
                                         "id": id,
                                         "answer_text": answer_text,
                                         "answer_start": answer_start,
                                         "is_impossible": is_impossible})
        df = pd.DataFrame.from_records(flat)
        return df

    def count(self, unit="questions"):
        c = 0
        for document in self.data:
            for paragraph in document["paragraphs"]:
                for question in paragraph["qas"]:
                    if unit == "questions":
                        c += 1
                    # Count no_answers
                    if len(question["answers"]) == 0:
                        if unit == "answers":
                            c += 1
                    # Count span answers
                    else:
                        for answer in question["answers"]:
                            if unit == "answers":
                                c += 1
        return c

    def df_to_data(self, df):
        df_answers = df.groupby(["title", "context", "question", "id",  "is_impossible"])


    @staticmethod
    def aggregate_answers(x):
        print()


    # def df_to_data(self, df):
    #     ret = []
    #     df_grouped_title = df.groupby("title")
    #     for title, df_title in tqdm(df_grouped_title):
    #         ret.append(
    #             {
    #                 "title": title,
    #                 "paragraphs": self._paragraphs_from_df(df_title)
    #             }
    #         )
    #     return ret
    #
    # def _paragraphs_from_df(self, df_title):
    #     ret = []
    #     df_grouped_context = df_title.groupby("context")
    #     for context, df_context in df_grouped_context:
    #         ret.append(
    #             {
    #                 "context": context,
    #                 "qas": self._qas_from_df(df_context)
    #             }
    #         )
    #     return ret
    #
    # def _qas_from_df(self, df_context):
    #     ret = []
    #     df_grouped_question = df_context.groupby(["question", "id", "is_impossible"])
    #     for (question, id, is_impossible), df_question in df_grouped_question:
    #         ret.append(
    #             {
    #                 "question": question,
    #                 "id": id,
    #                 "is_impossible": is_impossible,
    #                 "answers": self._answers_from_df(df_question)
    #             }
    #         )
    #     return ret
    #
    # def _answers_from_df(self, df_question):
    #     df_question = df_question[df_question["is_impossible"] == False]
    #     df_question = df_question[["answer_text", "answer_start"]]
    #     df_question = df_question.rename(columns={"answer_text": "text"})
    #     df_question["answer_start"] = df_question["answer_start"].astype(int)
    #     ret = df_question.to_dict("records")
    #     return ret

if __name__ == "__main__":
    sd = SquadData.from_file("../data/squad20/dev-v2.0.json")
    df = sd.to_df()
    data = sd.df_to_data(df)
    print()