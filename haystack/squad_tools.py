import json
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

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
        print("Converting data frame to squad format data")

        # Aggregate the answers of each question
        print("Aggregating the answers of each question")
        df_grouped_answers = df.groupby(["title", "context", "question", "id",  "is_impossible"])
        df_aggregated_answers = df[["title", "context", "question", "id",  "is_impossible"]].drop_duplicates().reset_index()
        answers = df_grouped_answers.progress_apply(self.aggregate_answers).rename("answers")
        answers = pd.DataFrame(answers).reset_index()
        df_aggregated_answers = pd.merge(df_aggregated_answers, answers)

        # Aggregate the questions of each passage
        print("Aggregating the questions of each paragraphs of each document")
        df_grouped_questions = df_aggregated_answers.groupby(["title", "context"])
        df_aggregated_questions = df[["title", "context"]].drop_duplicates().reset_index()
        questions = df_grouped_questions.progress_apply(self.aggregate_questions).rename("qas")
        questions = pd.DataFrame(questions).reset_index()
        df_aggregated_questions = pd.merge(df_aggregated_questions, questions)

        print("Aggregating the paragraphs of each document")
        df_grouped_paragraphs = df_aggregated_questions.groupby(["title"])
        df_aggregated_paragraphs = df[["title"]].drop_duplicates().reset_index()
        paragraphs = df_grouped_paragraphs.progress_apply(self.aggregate_passages).rename("paragraphs")
        paragraphs = pd.DataFrame(paragraphs).reset_index()
        df_aggregated_paragraphs = pd.merge(df_aggregated_paragraphs, paragraphs)

        df_aggregated_paragraphs = df_aggregated_paragraphs[["title", "paragraphs"]]
        ret = df_aggregated_paragraphs.to_dict("records")

        return ret

    @staticmethod
    def aggregate_passages(x):
        x = x[["context", "qas"]]
        ret = x.to_dict("records")
        return ret

    @staticmethod
    def aggregate_questions(x):
        x = x[["question", "id", "answers", "is_impossible"]]
        ret = x.to_dict("records")
        return ret

    @staticmethod
    def aggregate_answers(x):
        x = x[["answer_text", "answer_start"]]
        x = x.rename(columns={"answer_text": "text"})

        # Span anwser
        try:
            x["answer_start"] = x["answer_start"].astype(int)
            ret = x.to_dict("records")

        # No answer
        except ValueError:
            ret = []

        return ret

if __name__ == "__main__":
    sd = SquadData.from_file("../data/squad20/train-v2.0.json")
    df = sd.to_df()
    data_round_trip = sd.df_to_data(df)
    sd_round_trip = SquadData(data_round_trip)

    print(sd.count("answers"))
    print(sd_round_trip.count("answers"))

