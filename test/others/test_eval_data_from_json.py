import json
import os
import pytest
from haystack.document_stores import eval_data_from_json


@pytest.fixture
def temp_squad_file(tmp_path):
    temp_filename = tmp_path / "temp_squad_file.json"
    with open(temp_filename, "w", encoding="utf-8") as temp_file:
        json.dump(
            {
                "metadata": {
                    "dataset_version": "1.0",
                    "description": "This dataset contains questions and answers related to...",
                    "other_metadata_field": "value",
                },
                "data": [
                    {
                        "title": "Article Title",
                        "paragraphs": [
                            {
                                "context": "This is the context of the article.",
                                "qas": [
                                    {
                                        "question": "What is the SQuAD dataset?",
                                        "id": 0,
                                        "answers": [{"text": "This is the context", "answer_start": 0}],
                                        "annotator": "annotator0",
                                        "date": "2023-11-07",
                                    },
                                    {
                                        "question": "Another question?",
                                        "id": 1,
                                        "answers": [{"text": "This is the context of the article", "answer_start": 0}],
                                        "annotator": "annotator1",
                                        "date": "2023-12-09",
                                    },
                                ],
                            }
                        ],
                        "author": "Your Name",
                        "creation_date": "2023-11-14",
                    }
                ],
            },
            temp_file,
            indent=2,
        )
    return temp_filename


def test_eval_data_from_json(temp_squad_file):
    # Call the function with the temporary file
    docs, labels = eval_data_from_json(temp_squad_file)

    assert len(docs) == 1
    assert len(labels) == 2

    assert docs[0].content == "This is the context of the article."
    assert labels[0].query == "What is the SQuAD dataset?"
    assert labels[0].meta == {"annotator": "annotator0", "date": "2023-11-07"}

    assert labels[1].query == "Another question?"
    assert labels[1].meta == {"annotator": "annotator1", "date": "2023-12-09"}
