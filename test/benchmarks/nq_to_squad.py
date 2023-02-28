# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
DEEPSET DOCSTRING:

A modified version of the script from here:
https://github.com/google/retrieval-qa-eval/blob/main/nq_to_squad.py
Edits have been made by deepset in order to create a dev set for Haystack benchmarking.
Input should be the official NQ dev set (v1.0-simplified-nq-dev-all.jsonl.gz)

Expected numbers are:
Converted 7830 NQ records into 5678 SQuAD records.
Removed samples: yes/no: 177 multi_short: 648 non_para 1192 long_ans_only: 130 errors: 5
Removed annotations: long_answer: 4610 short_answer: 953 no_answer: ~1006
where:
multi_short - annotations where there are multiple disjoint short answers
non_para - where the annotation occurs in an html element that is not a paragraph


ORIGINAL DOCSTRING:

Convert the Natural Questions dataset into SQuAD JSON format.

To use this utility, first follow the directions at the URL below to download
the complete training dataset.

    https://ai.google.com/research/NaturalQuestions/download

Next, run this program, specifying the data you wish to convert. For instance,
the invocation:

    python nq_to_squad.py\
        --data_pattern=/usr/local/data/tnq/v1.0/train/*.gz\
        --output_file=/usr/local/data/tnq/v1.0/train.json

will process all training data and write the results into `train.json`. This
file can, in turn, be provided to squad_eval.py using the --squad argument.
"""

import argparse
import glob
import gzip
import json
import logging
import os
import re


logger = logging.getLogger(__name__)


# Dropped samples
n_yn = 0
n_ms = 0
n_non_p = 0
n_long_ans_only = 0
n_error = 0

# Dropped annotations
n_long_ans = 0
n_no_ans = 0
n_short = 0


def clean_text(start_token, end_token, doc_tokens, doc_bytes, ignore_final_whitespace=True):
    """Remove HTML tags from a text span and reconstruct proper spacing."""
    text = ""
    for index in range(start_token, end_token):
        token = doc_tokens[index]
        if token["html_token"]:
            continue
        text += token["token"]
        # Add a single space between two tokens iff there is at least one
        # whitespace character between them (outside of an HTML tag). For example:
        #
        #   token1 token2                           ==> Add space.
        #   token1</B> <B>token2                    ==> Add space.
        #   token1</A>token2                        ==> No space.
        #   token1<A href="..." title="...">token2  ==> No space.
        #   token1<SUP>2</SUP>token2                ==> No space.
        next_token = token
        last_index = end_token if ignore_final_whitespace else end_token + 1
        for next_token in doc_tokens[index + 1 : last_index]:
            if not next_token["html_token"]:
                break
        chars = doc_bytes[token["end_byte"] : next_token["start_byte"]].decode("utf-8")
        # Since some HTML tags are missing from the token list, we count '<' and
        # '>' to detect if we're inside a tag.
        unclosed_brackets = 0
        for char in chars:
            if char == "<":
                unclosed_brackets += 1
            elif char == ">":
                unclosed_brackets -= 1
            elif unclosed_brackets == 0 and re.match(r"\s", char):
                # Add a single space after this token.
                text += " "
                break
    return text


def get_anno_type(annotation):
    long_answer = annotation["long_answer"]
    short_answers = annotation["short_answers"]
    yes_no_answer = annotation["yes_no_answer"]

    if len(short_answers) > 1:
        return "multi_short"
    elif yes_no_answer != "NONE":
        return yes_no_answer
    elif len(short_answers) == 1:
        return "short_answer"
    elif len(short_answers) == 0:
        if long_answer["start_token"] == -1:
            return "no_answer"
        else:
            return "long_answer"


def reduce_annotations(anno_types, answers):
    """
    In cases where there is annotator disagreement, this fn picks either only the short_answers or only the no_answers,
    depending on which is more numerous, with a bias towards picking short_answers.

    Note: By this stage, all long_answer annotations and all samples with yes/no answer have been removed.
    This leaves just no_answer and short_answers"""
    for at in set(anno_types):
        assert at in ("no_answer", "short_answer")
    if anno_types.count("short_answer") >= anno_types.count("no_answer"):
        majority = "short_answer"
        is_impossible = False
    else:
        majority = "no_answer"
        is_impossible = True
    answers = [a for at, a in zip(anno_types, answers) if at == majority]
    reduction = len(anno_types) - len(answers)
    assert reduction < 3
    if not is_impossible:
        global n_no_ans
        n_no_ans += reduction
    else:
        global n_short
        n_short += reduction
        answers = []
    return answers, is_impossible


def nq_to_squad(record):
    """Convert a Natural Questions record to SQuAD format."""

    doc_bytes = record["document_html"].encode("utf-8")
    doc_tokens = record["document_tokens"]

    question_text = record["question_text"]
    question_text = question_text[0].upper() + question_text[1:] + "?"

    answers = []
    anno_types = []
    for annotation in record["annotations"]:
        anno_type = get_anno_type(annotation)
        long_answer = annotation["long_answer"]
        short_answers = annotation["short_answers"]

        if anno_type.lower() in ["yes", "no"]:
            global n_yn
            n_yn += 1
            return

        # Skip examples that don't have exactly one short answer.
        # Note: Consider including multi-span short answers.
        if anno_type == "multi_short":
            global n_ms
            n_ms += 1
            return

        elif anno_type == "short_answer":
            short_answer = short_answers[0]
            # Skip examples corresponding to HTML blocks other than <P>.
            long_answer_html_tag = doc_tokens[long_answer["start_token"]]["token"]
            if long_answer_html_tag != "<P>":
                global n_non_p
                n_non_p += 1
                return
            answer = clean_text(short_answer["start_token"], short_answer["end_token"], doc_tokens, doc_bytes)
            before_answer = clean_text(
                0, short_answer["start_token"], doc_tokens, doc_bytes, ignore_final_whitespace=False
            )

        elif anno_type == "no_answer":
            answer = ""
            before_answer = ""

        # Throw out long answer annotations
        elif anno_type == "long_answer":
            global n_long_ans
            n_long_ans += 1
            continue

        anno_types.append(anno_type)
        answer = {"answer_start": len(before_answer), "text": answer}
        answers.append(answer)

    if len(answers) == 0:
        global n_long_ans_only
        n_long_ans_only += 1
        return

    answers, is_impossible = reduce_annotations(anno_types, answers)

    paragraph = clean_text(0, len(doc_tokens), doc_tokens, doc_bytes)

    return {
        "title": record["document_title"],
        "paragraphs": [
            {
                "context": paragraph,
                "qas": [
                    {
                        "answers": answers,
                        "id": record["example_id"],
                        "question": question_text,
                        "is_impossible": is_impossible,
                    }
                ],
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert the Natural Questions to SQuAD JSON format.")
    parser.add_argument(
        "--data_pattern",
        dest="data_pattern",
        help=("A file pattern to match the Natural Questions " "dataset."),
        metavar="PATTERN",
        required=True,
    )
    parser.add_argument(
        "--version", dest="version", help="The version label in the output file.", metavar="LABEL", default="nq-train"
    )
    parser.add_argument(
        "--output_file",
        dest="output_file",
        help="The name of the SQuAD JSON formatted output file.",
        metavar="FILE",
        default="nq_as_squad.json",
    )
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    records = 0
    nq_as_squad = {"version": args.version, "data": []}

    for file in sorted(glob.iglob(args.data_pattern)):
        logger.info("opening %s", file)
        with gzip.GzipFile(file, "r") as f:
            for line in f:
                records += 1
                nq_record = json.loads(line)
                try:
                    squad_record = nq_to_squad(nq_record)
                except:
                    squad_record = None
                    global n_error
                    n_error += 1
                if squad_record:
                    nq_as_squad["data"].append(squad_record)
                if records % 100 == 0:
                    logger.info("processed %s records", records)
    print("Converted %s NQ records into %s SQuAD records." % (records, len(nq_as_squad["data"])))
    print(
        f"Removed samples: yes/no: {n_yn} multi_short: {n_ms} non_para {n_non_p} long_ans_only: {n_long_ans_only} errors: {n_error}"
    )
    print(f"Removed annotations: long_answer: {n_long_ans} short_answer: {n_short} no_answer: ~{n_no_ans}")

    with open(args.output_file, "w") as f:
        json.dump(nq_as_squad, f, indent=4)


if __name__ == "__main__":
    main()
