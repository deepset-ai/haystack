import json
from collections import defaultdict
import logging
import pprint
from typing import Dict, Any
import pandas as pd
import re
import warnings
import string
from haystack.database.sql import DocumentORM
from haystack.schemas import SquadSchema, paragraphs, qas, answer, data
logger = logging.getLogger(__name__)


def print_answers(results: dict, details: str = "all"):
    answers = results["answers"]
    pp = pprint.PrettyPrinter(indent=4)
    if details != "all":
        if details == "minimal":
            keys_to_keep = set(["answer", "context"])
        elif details == "medium":
            keys_to_keep = set(["answer", "context", "score"])
        else:
            keys_to_keep = answers.keys()

        # filter the results
        filtered_answers = []
        for ans in answers:
            filtered_answers.append({k: ans[k] for k in keys_to_keep})
        pp.pprint(filtered_answers)
    else:
        pp.pprint(results)


def convert_labels_to_squad(labels_file: str):
    """
    Convert the export from the labeling UI to SQuAD format for training.

    :param labels_file: path for export file from the labeling tool
    :return:
    """
    with open(labels_file) as label_file:
        labels = json.load(label_file)

    labels_grouped_by_documents = defaultdict(list)
    for label in labels:
        labels_grouped_by_documents[label["document_id"]].append(label)

    labels_in_squad_format = {"data": []}  # type: Dict[str, Any]
    for document_id, labels in labels_grouped_by_documents.items():
        qas = []
        for label in labels:
            doc = DocumentORM.query.get(label["document_id"])

            assert (
                doc.text[label["start_offset"] : label["end_offset"]]
                == label["selected_text"]
            )

            qas.append(
                {
                    "question": label["question"],
                    "id": label["id"],
                    "question_id": label["question_id"],
                    "answers": [
                        {
                            "text": label["selected_text"],
                            "answer_start": label["start_offset"],
                            "labeller_id": label["labeler_id"],
                        }
                    ],
                    "is_impossible": False,
                }
            )

        squad_format_label = {
            "paragraphs": [
                {"qas": qas, "context": doc.text, "document_id": document_id}
            ]
        }

        labels_in_squad_format["data"].append(squad_format_label)

    with open("labels_in_squad_format.json", "w+") as outfile:
        json.dump(labels_in_squad_format, outfile)

def find_answer_start(answer: str, text: str):
    """
    Get index of beginning of answer in text

    :param text: string where 'answer' is to be searched in
    :param answer: substring to be searched in 'text'
    :return: list of indices of occurrences of answer
    """
    # escape all (
    special_chars = {"(": "\(", ")": "\)", "[": "\[", "]":"\]", "+":"\+", "*":"\*"}
    table = str.maketrans(special_chars)
    answer = answer.translate(table)

    answers = [m.start() for m in re.finditer(answer.lower(), text.lower())]
    if not answers:
        warnings.warn("No answer found in context!")
    if answers[1:]:
        warnings.warn("More than one occurrence of answer found. Treating all occurrences as different answers.")
    return answers



def convert_df_to_squad(dataframe):
    """
    Convert pandas data-frame to squad json

    :param dataframe: Pandas dataframe where each row represents one question-context sample with other relevant info
                    columns in the dataframe:
                        question: question string: str
                        answers: list of answers: [str]
                        is_impossible: is answer absent: Optional[bool]
                        context: context passage: str
                        passage_id: unique identification number for a context passage: int
                        title: title of the context passage: str
    :return: json in Squad Format
    """
    squad_data = []

    # Group rows with same context 'title'
    for title, all_paragraphs in dataframe.groupby(['title']):
        paras_in_title = []

        # Group rows with same context passages identified with 'passage_id'
        for para_id, para in all_paragraphs.groupby(['passage_id']):
            ques_ans_in_paragraph = []

            # create Squad 'qas'
            for question, answers in zip(para['question'], para['answers']):

                # group Squad 'answers' within a list of answers
                ans_ = [answer(text=ans, answer_start=index) for ans in answers
                        for index in find_answer_start(ans, para.context.iloc[0])]
                # qas
                if ans_:
                    ques_ans_in_paragraph.append(qas(question=question, answers=ans_))
                else:
                    warnings.warn(f"No answer span found for question \"{question}\"! Skipping qas!")

            # Group all passages with same title within a list of passages to create Squad 'passages'
            if ques_ans_in_paragraph:
                paras_in_title.append(paragraphs(qas=ques_ans_in_paragraph,
                                                 context=para.context.iloc[0],
                                                 passage_id=para_id))
            else:
                warnings.warn(f"No answer span found for paragraph \"{para.context.iloc[0]}\"! Skipping paragraph")

        # squad 'data': list of articles grouped by title
        if paras_in_title:
            squad_data.append(data(title=title, paragraphs=paras_in_title))
    if not squad_data:
        raise ValueError("squad data is empty! Check whether answers exist in context!")
    return SquadSchema(data=squad_data).json()

def convert_dpr_to_squad(input_file: str, output_file: str):
    """
    Convert a Dense Passage Retrieval (DPR) json file to squad-format json and write to output_file

    :param input_file: path to json file in DPR data format
    :param output_file: path to output json file
    :return Squad json
    """
    # convert json to pandas dataframe
    json_data = json.load(open(input_file))
    samples_dataframe = pd.json_normalize(json_data)

    col_name_mapping = {'questions': 'question', 'answers': 'answers', 'positive_contexts': 'positive_ctxs',
                        'psg_id': 'passage_id', 'text': 'context'}

    # Remove negative and hard negative context
    samples_dataframe = samples_dataframe.rename(
        columns=col_name_mapping)[['question', 'answers', 'positive_ctxs']]

    # remove samples without positive context
    samples_dataframe = samples_dataframe[samples_dataframe['positive_ctxs'].apply(lambda x: len(x)) > 0]

    # Only keep the 1 positive context
    pos_ctxs = samples_dataframe['positive_ctxs'].transform(lambda x: x[0]).apply(pd.Series)

    pos_ctxs = pos_ctxs.rename(columns=col_name_mapping)[["title", "context", "passage_id"]]
    df = pd.concat([samples_dataframe[['question', 'answers']], pos_ctxs], axis=1)

    # convert data frame to SQuAD format
    out_json = convert_df_to_squad(df)

    # write output json
    if output_file:
        with open(output_file, "w") as f_out:
            json.dump(json.loads(out_json), f_out, sort_keys=True, indent=4)
    return json.loads(out_json)