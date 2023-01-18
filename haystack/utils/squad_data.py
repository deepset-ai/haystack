from typing import List

import logging
import json
import random
import pandas as pd
from tqdm.auto import tqdm
import mmh3

from haystack.schema import Document, Label, Answer
from haystack.modeling.data_handler.processor import _read_squad_file


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

tqdm.pandas()


COLUMN_NAMES = ["title", "context", "question", "id", "answer_text", "answer_start", "is_impossible"]


class SquadData:
    """
    This class is designed to manipulate data that is in SQuAD format
    """

    def __init__(self, squad_data):
        """
        :param squad_data: SQuAD format data, either as a dictionary with a `data` key, or just a list of SQuAD documents.
        """
        if type(squad_data) == dict:
            self.version = squad_data.get("version")
            self.data = squad_data["data"]
        elif type(squad_data) == list:
            self.version = None
            self.data = squad_data
        self.df = self.to_df(self.data)

    def merge_from_file(self, filename: str):
        """Merge the contents of a JSON file in the SQuAD format with the data stored in this object."""
        new_data = json.load(open(filename))["data"]
        self.merge(new_data)

    def merge(self, new_data: List):
        """
        Merge data in SQuAD format with the data stored in this object.
        :param new_data: A list of SQuAD document data.
        """
        df_new = self.to_df(new_data)
        self.df = pd.concat([df_new, self.df])
        self.data = self.df_to_data(self.df)

    @classmethod
    def from_file(cls, filename: str):
        """
        Create a SquadData object by providing the name of a JSON file in the SQuAD format.
        """
        with open(filename) as f:
            data = json.load(f)
        return cls(data)

    def save(self, filename: str):
        """
        Write the data stored in this object to a JSON file.
        """
        with open(filename, "w") as f:
            squad_data = {"version": self.version, "data": self.data}
            json.dump(squad_data, f, indent=2)

    def to_dpr_dataset(self):
        raise NotImplementedError(
            "SquadData.to_dpr_dataset() not yet implemented. "
            "For now, have a look at the script at haystack/retriever/squad_to_dpr.py"
        )

    def to_document_objs(self):
        """
        Export all paragraphs stored in this object to haystack.Document objects.
        """
        df_docs = self.df[["title", "context"]]
        df_docs = df_docs.drop_duplicates()
        record_dicts = df_docs.to_dict("records")
        documents = [Document(content=rd["context"], id=rd["title"]) for rd in record_dicts]
        return documents

    def to_label_objs(self, answer_type="generative"):
        """Export all labels stored in this object to haystack.Label objects"""
        df_labels = self.df[["id", "question", "answer_text", "answer_start", "context", "document_id"]]
        record_dicts = df_labels.to_dict("records")
        labels = [
            Label(
                query=record["question"],
                answer=Answer(answer=record["answer_text"], answer_type=answer_type),
                is_correct_answer=True,
                is_correct_document=True,
                id=record["id"],
                origin=record.get("origin", "gold-label"),
                document=Document(content=record.get("context"), id=str(record["document_id"])),
            )
            for record in record_dicts
        ]
        return labels

    @staticmethod
    def to_df(data):
        """Convert a list of SQuAD document dictionaries into a pandas dataframe (each row is one annotation)."""
        flat = []
        for document in data:
            title = document.get("title", "")
            for paragraph in document["paragraphs"]:
                context = paragraph["context"]
                document_id = paragraph.get("document_id", "{:02x}".format(mmh3.hash128(str(context), signed=False)))
                for question in paragraph["qas"]:
                    q = question["question"]
                    id = question["id"]
                    is_impossible = question.get("is_impossible", False)
                    # For no_answer samples
                    if len(question["answers"]) == 0:
                        flat.append(
                            {
                                "title": title,
                                "context": context,
                                "question": q,
                                "id": id,
                                "answer_text": "",
                                "answer_start": None,
                                "is_impossible": is_impossible,
                                "document_id": document_id,
                            }
                        )
                    # For span answer samples
                    else:
                        for answer in question["answers"]:
                            answer_text = answer["text"]
                            answer_start = answer["answer_start"]
                            flat.append(
                                {
                                    "title": title,
                                    "context": context,
                                    "question": q,
                                    "id": id,
                                    "answer_text": answer_text,
                                    "answer_start": answer_start,
                                    "is_impossible": is_impossible,
                                    "document_id": document_id,
                                }
                            )
        df = pd.DataFrame.from_records(flat)
        return df

    def count(self, unit="questions"):
        """
        Count the samples in the data. Choose a unit: "paragraphs", "questions", "answers", "no_answers", "span_answers".
        """
        c = 0
        for document in self.data:
            for paragraph in document["paragraphs"]:
                if unit == "paragraphs":
                    c += 1
                for question in paragraph["qas"]:
                    if unit == "questions":
                        c += 1
                    # Count no_answers
                    if len(question["answers"]) == 0:
                        if unit in ["answers", "no_answers"]:
                            c += 1
                    # Count span answers
                    else:
                        for _ in question["answers"]:
                            if unit in ["answers", "span_answers"]:
                                c += 1
        return c

    @classmethod
    def df_to_data(cls, df):
        """
        Convert a data frame into the SQuAD format data (list of SQuAD document dictionaries).
        """
        logger.info("Converting data frame to squad format data")

        # Aggregate the answers of each question
        logger.info("Aggregating the answers of each question")
        df_grouped_answers = df.groupby(["title", "context", "question", "id", "is_impossible"])
        df_aggregated_answers = (
            df[["title", "context", "question", "id", "is_impossible"]].drop_duplicates().reset_index()
        )
        answers = df_grouped_answers.progress_apply(cls._aggregate_answers).rename("answers")
        answers = pd.DataFrame(answers).reset_index()
        df_aggregated_answers = pd.merge(df_aggregated_answers, answers)

        # Aggregate the questions of each passage
        logger.info("Aggregating the questions of each paragraphs of each document")
        df_grouped_questions = df_aggregated_answers.groupby(["title", "context"])
        df_aggregated_questions = df[["title", "context"]].drop_duplicates().reset_index()
        questions = df_grouped_questions.progress_apply(cls._aggregate_questions).rename("qas")
        questions = pd.DataFrame(questions).reset_index()
        df_aggregated_questions = pd.merge(df_aggregated_questions, questions)

        logger.info("Aggregating the paragraphs of each document")
        df_grouped_paragraphs = df_aggregated_questions.groupby(["title"])
        df_aggregated_paragraphs = df[["title"]].drop_duplicates().reset_index()
        paragraphs = df_grouped_paragraphs.progress_apply(cls._aggregate_passages).rename("paragraphs")
        paragraphs = pd.DataFrame(paragraphs).reset_index()
        df_aggregated_paragraphs = pd.merge(df_aggregated_paragraphs, paragraphs)

        df_aggregated_paragraphs = df_aggregated_paragraphs[["title", "paragraphs"]]
        ret = df_aggregated_paragraphs.to_dict("records")

        return ret

    @staticmethod
    def _aggregate_passages(x):
        x = x[["context", "qas"]]
        ret = x.to_dict("records")
        return ret

    @staticmethod
    def _aggregate_questions(x):
        x = x[["question", "id", "answers", "is_impossible"]]
        ret = x.to_dict("records")
        return ret

    @staticmethod
    def _aggregate_answers(x):
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

    def set_data(self, data):
        self.data = data
        self.df = self.to_df(data)

    def sample_questions(self, n):
        """
        Return a sample of n questions in the SQuAD format (a list of SQuAD document dictionaries).
        Note that if the same question is asked on multiple different passages, this function treats that
        as a single question.
        """
        all_questions = self.get_all_questions()
        sampled_questions = random.sample(all_questions, n)
        df_sampled = self.df[self.df["question"].isin(sampled_questions)]
        return self.df_to_data(df_sampled)

    def get_all_paragraphs(self):
        """
        Return all paragraph strings.
        """
        return self.df["context"].unique().tolist()

    def get_all_questions(self):
        """
        Return all question strings. Note that if the same question appears for different paragraphs, this function returns it multiple times.
        """
        df_questions = self.df[["title", "context", "question"]]
        df_questions = df_questions.drop_duplicates()
        questions = df_questions["question"].tolist()
        return questions

    def get_all_document_titles(self):
        """Return all document title strings."""
        return self.df["title"].unique().tolist()


if __name__ == "__main__":
    # Download the SQuAD dataset if it isn't at target directory
    _read_squad_file("../data/squad20/train-v2.0.json")

    filename1 = "../data/squad20/train-v2.0.json"
    filename2 = "../data/squad20/dev-v2.0.json"

    # Load file1 and take a sample of 10000 questions
    sd = SquadData.from_file(filename1)
    sample1 = sd.sample_questions(n=10000)

    # Set sd to now contain the sample of 10000 questions
    sd.set_data(sample1)

    # Merge sd with file2 and take a sample of 100 questions
    sd.merge_from_file(filename2)
    sample2 = sd.sample_questions(n=100)
    sd.set_data(sample2)

    # Save this sample of 100
    sd.save("../data/squad20/sample.json")

    paragraphs = sd.get_all_paragraphs()
    questions = sd.get_all_questions()
    titles = sd.get_all_document_titles()

    documents = sd.to_document_objs()
    labels = sd.to_label_objs()

    n_qs = sd.count(unit="questions")
    n_as = sd.count(unit="no_answers")
    n_ps = sd.count(unit="paragraphs")

    print(n_qs)
    print(n_as)
    print(n_ps)
