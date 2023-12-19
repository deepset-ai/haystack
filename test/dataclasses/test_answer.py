from pandas import DataFrame

from haystack.dataclasses.answer import Answer, ExtractedAnswer, ExtractedTableAnswer, GeneratedAnswer
from haystack.dataclasses.document import Document


class TestExtractedAnswer:
    def test_init(self):
        answer = ExtractedAnswer(
            data="42",
            query="What is the answer?",
            document=Document(content="I thought a lot about this. The answer is 42."),
            context="The answer is 42.",
            score=1.0,
            document_offset=ExtractedAnswer.Span(42, 44),
            context_offset=ExtractedAnswer.Span(14, 16),
            meta={"meta_key": "meta_value"},
        )
        assert answer.data == "42"
        assert answer.query == "What is the answer?"
        assert answer.document == Document(content="I thought a lot about this. The answer is 42.")
        assert answer.context == "The answer is 42."
        assert answer.score == 1.0
        assert answer.document_offset == ExtractedAnswer.Span(42, 44)
        assert answer.context_offset == ExtractedAnswer.Span(14, 16)
        assert answer.meta == {"meta_key": "meta_value"}

    def test_protocol(self):
        answer = ExtractedAnswer(
            data="42",
            query="What is the answer?",
            document=Document(content="I thought a lot about this. The answer is 42."),
            context="The answer is 42.",
            score=1.0,
            document_offset=ExtractedAnswer.Span(42, 44),
            context_offset=ExtractedAnswer.Span(14, 16),
            meta={"meta_key": "meta_value"},
        )
        assert isinstance(answer, Answer)

    def test_to_dict(self):
        document = Document(content="I thought a lot about this. The answer is 42.")
        answer = ExtractedAnswer(
            data="42",
            query="What is the answer?",
            document=document,
            context="The answer is 42.",
            score=1.0,
            document_offset=ExtractedAnswer.Span(42, 44),
            context_offset=ExtractedAnswer.Span(14, 16),
            meta={"meta_key": "meta_value"},
        )
        assert answer.to_dict() == {
            "type": "haystack.dataclasses.answer.ExtractedAnswer",
            "init_parameters": {
                "data": "42",
                "query": "What is the answer?",
                "document": document.to_dict(flatten=False),
                "context": "The answer is 42.",
                "score": 1.0,
                "document_offset": {"start": 42, "end": 44},
                "context_offset": {"start": 14, "end": 16},
                "meta": {"meta_key": "meta_value"},
            },
        }

    def test_from_dict(self):
        answer = ExtractedAnswer.from_dict(
            {
                "type": "haystack.dataclasses.answer.ExtractedAnswer",
                "init_parameters": {
                    "data": "42",
                    "query": "What is the answer?",
                    "document": {
                        "id": "8f800a524b139484fc719ecc35f971a080de87618319bc4836b784d69baca57f",
                        "content": "I thought a lot about this. The answer is 42.",
                    },
                    "context": "The answer is 42.",
                    "score": 1.0,
                    "document_offset": {"start": 42, "end": 44},
                    "context_offset": {"start": 14, "end": 16},
                    "meta": {"meta_key": "meta_value"},
                },
            }
        )
        assert answer.data == "42"
        assert answer.query == "What is the answer?"
        assert answer.document == Document(
            id="8f800a524b139484fc719ecc35f971a080de87618319bc4836b784d69baca57f",
            content="I thought a lot about this. The answer is 42.",
        )
        assert answer.context == "The answer is 42."
        assert answer.score == 1.0
        assert answer.document_offset == ExtractedAnswer.Span(42, 44)
        assert answer.context_offset == ExtractedAnswer.Span(14, 16)
        assert answer.meta == {"meta_key": "meta_value"}


class TestExtractedTableAnswer:
    def test_init(self):
        answer = ExtractedTableAnswer(
            data="42",
            query="What is the answer?",
            document=Document(dataframe=DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 42]})),
            context=DataFrame({"col3": [5, 42]}),
            score=1.0,
            document_cells=[ExtractedTableAnswer.Cell(1, 2)],
            context_cells=[ExtractedTableAnswer.Cell(1, 0)],
            meta={"meta_key": "meta_value"},
        )
        assert answer.data == "42"
        assert answer.query == "What is the answer?"
        assert answer.document == Document(dataframe=DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 42]}))
        assert answer.context.equals(DataFrame({"col3": [5, 42]}))
        assert answer.score == 1.0
        assert answer.document_cells == [ExtractedTableAnswer.Cell(1, 2)]
        assert answer.context_cells == [ExtractedTableAnswer.Cell(1, 0)]
        assert answer.meta == {"meta_key": "meta_value"}

    def test_protocol(self):
        answer = ExtractedTableAnswer(
            data="42",
            query="What is the answer?",
            document=Document(dataframe=DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 42]})),
            context=DataFrame({"col3": [5, 42]}),
            score=1.0,
            document_cells=[ExtractedTableAnswer.Cell(1, 2)],
            context_cells=[ExtractedTableAnswer.Cell(1, 0)],
            meta={"meta_key": "meta_value"},
        )
        assert isinstance(answer, Answer)

    def test_to_dict(self):
        document = Document(dataframe=DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 42]}))
        answer = ExtractedTableAnswer(
            data="42",
            query="What is the answer?",
            document=document,
            context=DataFrame({"col3": [5, 42]}),
            score=1.0,
            document_cells=[ExtractedTableAnswer.Cell(1, 2)],
            context_cells=[ExtractedTableAnswer.Cell(1, 0)],
            meta={"meta_key": "meta_value"},
        )

        assert answer.to_dict() == {
            "type": "haystack.dataclasses.answer.ExtractedTableAnswer",
            "init_parameters": {
                "data": "42",
                "query": "What is the answer?",
                "document": document.to_dict(flatten=False),
                "context": DataFrame({"col3": [5, 42]}).to_json(),
                "score": 1.0,
                "document_cells": [{"row": 1, "column": 2}],
                "context_cells": [{"row": 1, "column": 0}],
                "meta": {"meta_key": "meta_value"},
            },
        }

    def test_from_dict(self):
        answer = ExtractedTableAnswer.from_dict(
            {
                "type": "haystack.dataclasses.answer.ExtractedTableAnswer",
                "init_parameters": {
                    "data": "42",
                    "query": "What is the answer?",
                    "document": {
                        "id": "3b13a0d56a3697e27a874fcb621911c83c59388dec213909e9e40d5d9f0affed",
                        "dataframe": '{"col1":{"0":1,"1":2},"col2":{"0":3,"1":4},"col3":{"0":5,"1":42}}',
                    },
                    "context": '{"col3":{"0":5,"1":42}}',
                    "score": 1.0,
                    "document_cells": [{"row": 1, "column": 2}],
                    "context_cells": [{"row": 1, "column": 0}],
                    "meta": {"meta_key": "meta_value"},
                },
            }
        )
        assert answer.data == "42"
        assert answer.query == "What is the answer?"
        assert answer.document == Document(
            id="3b13a0d56a3697e27a874fcb621911c83c59388dec213909e9e40d5d9f0affed",
            dataframe=DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 42]}),
        )
        assert answer.context.equals(DataFrame({"col3": [5, 42]}))
        assert answer.score == 1.0
        assert answer.document_cells == [ExtractedTableAnswer.Cell(1, 2)]
        assert answer.context_cells == [ExtractedTableAnswer.Cell(1, 0)]
        assert answer.meta == {"meta_key": "meta_value"}


class TestGeneratedAnswer:
    def test_init(self):
        answer = GeneratedAnswer(
            data="42",
            query="What is the answer?",
            documents=[
                Document(id="1", content="The answer is 42."),
                Document(id="2", content="I believe the answer is 42."),
                Document(id="3", content="42 is definitely the answer."),
            ],
            meta={"meta_key": "meta_value"},
        )
        assert answer.data == "42"
        assert answer.query == "What is the answer?"
        assert answer.documents == [
            Document(id="1", content="The answer is 42."),
            Document(id="2", content="I believe the answer is 42."),
            Document(id="3", content="42 is definitely the answer."),
        ]
        assert answer.meta == {"meta_key": "meta_value"}

    def test_protocol(self):
        answer = GeneratedAnswer(
            data="42",
            query="What is the answer?",
            documents=[
                Document(id="1", content="The answer is 42."),
                Document(id="2", content="I believe the answer is 42."),
                Document(id="3", content="42 is definitely the answer."),
            ],
            meta={"meta_key": "meta_value"},
        )
        assert isinstance(answer, Answer)

    def test_to_dict(self):
        documents = [
            Document(id="1", content="The answer is 42."),
            Document(id="2", content="I believe the answer is 42."),
            Document(id="3", content="42 is definitely the answer."),
        ]
        answer = GeneratedAnswer(
            data="42", query="What is the answer?", documents=documents, meta={"meta_key": "meta_value"}
        )
        assert answer.to_dict() == {
            "type": "haystack.dataclasses.answer.GeneratedAnswer",
            "init_parameters": {
                "data": "42",
                "query": "What is the answer?",
                "documents": [d.to_dict(flatten=False) for d in documents],
                "meta": {"meta_key": "meta_value"},
            },
        }

    def test_from_dict(self):
        answer = GeneratedAnswer.from_dict(
            {
                "type": "haystack.dataclasses.answer.GeneratedAnswer",
                "init_parameters": {
                    "data": "42",
                    "query": "What is the answer?",
                    "documents": [
                        {"id": "1", "content": "The answer is 42."},
                        {"id": "2", "content": "I believe the answer is 42."},
                        {"id": "3", "content": "42 is definitely the answer."},
                    ],
                    "meta": {"meta_key": "meta_value"},
                },
            }
        )
        assert answer.data == "42"
        assert answer.query == "What is the answer?"
        assert answer.documents == [
            Document(id="1", content="The answer is 42."),
            Document(id="2", content="I believe the answer is 42."),
            Document(id="3", content="42 is definitely the answer."),
        ]
        assert answer.meta == {"meta_key": "meta_value"}
