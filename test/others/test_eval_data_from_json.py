import unittest
import json
from haystack.document_stores import eval_data_from_json


class TestEvalDataFromJSON(unittest.TestCase):
    def test_eval_data_from_json(self):
        # Create a temporary SQuAD-style JSON file for testing
        temp_filename = "temp_squad_file.json"
        with open(temp_filename, "w", encoding="utf-8") as temp_file:
            # Use the provided example JSON data
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
                                            "answers": [
                                                {"text": "This is the context of the article", "answer_start": 0}
                                            ],
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
                indent=2,  # Adjust indentation for readability
            )

        # Call the function with the temporary file
        docs, labels = eval_data_from_json(temp_filename)

        # Add your assertions based on the structure of your data
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(labels), 2)

        # Assuming that your Document and Label classes have attributes like 'text', 'question', 'answer', etc.
        self.assertEqual(docs[0].content, "This is the context of the article.")
        self.assertEqual(labels[0].query, "What is the SQuAD dataset?")
        self.assertEqual(labels[0].meta, {"annotator": "annotator0", "date": "2023-11-07"})

        self.assertEqual(labels[1].query, "Another question?")
        self.assertEqual(labels[1].meta, {"annotator": "annotator1", "date": "2023-12-09"})

        # Clean up: Remove the temporary file
        import os

        os.remove(temp_filename)


if __name__ == "__main__":
    unittest.main()
