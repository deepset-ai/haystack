import pytest

from haystack import Document, Answer, ExtractedAnswer, GeneratedAnswer


def test_answer_to_dict():
    answer = Answer(data="Giorgio and I", query="Who lives in Rome?", metadata={"text": "test metadata"})

    assert answer.to_dict() == {"data": "Giorgio and I", "query": "Who lives in Rome?", "text": "test metadata"}


def test_answer_to_dict_without_flattening():
    answer = Answer(data="Giorgio and I", query="Who lives in Rome?", metadata={"text": "test metadata"})

    assert answer.to_dict(flatten=False) == {
        "data": "Giorgio and I",
        "query": "Who lives in Rome?",
        "metadata": {"text": "test metadata"},
    }


def test_answer_from_dict():
    data = {"data": "Giorgio and I", "query": "Who lives in Rome?", "text": "test metadata"}

    assert Answer.from_dict(data) == Answer(
        data="Giorgio and I", query="Who lives in Rome?", metadata={"text": "test metadata"}
    )


def test_answer_from_dict_without_flattening():
    data = {"data": "Giorgio and I", "query": "Who lives in Rome?", "metadata": {"text": "test metadata"}}

    assert Answer.from_dict(data) == Answer(
        data="Giorgio and I", query="Who lives in Rome?", metadata={"text": "test metadata"}
    )


def test_answer_from_dict_with_flat_and_non_flat_meta():
    with pytest.raises(ValueError, match="Pass either the 'metadata' parameter or flattened metadata keys"):
        Answer.from_dict(
            {
                "data": "Giorgio and I",
                "query": "Who lives in Rome?",
                "metadata": {"text": "test metadata"},
                "text": "test metadata other",
            }
        )


def test_extracted_answer_to_dict():
    extracted_answer = ExtractedAnswer(
        data="Giorgio and I",
        query="Who lives in Rome?",
        metadata={"text": "test metadata"},
        document=Document(
            id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            content="My name is Giorgio and I live in Rome.",
            score=0.33144005810482535,
        ),
        probability=0.7661304473876953,
        start=11,
        end=24,
    )

    assert extracted_answer.to_dict() == {
        "data": "Giorgio and I",
        "query": "Who lives in Rome?",
        "document": {
            "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            "content": "My name is Giorgio and I live in Rome.",
            "dataframe": None,
            "blob": None,
            "score": 0.33144005810482535,
            "embedding": None,
        },
        "probability": 0.7661304473876953,
        "start": 11,
        "end": 24,
        "text": "test metadata",
    }


def test_extracted_answer_to_dict_without_flattening():
    extracted_answer = ExtractedAnswer(
        data="Giorgio and I",
        query="Who lives in Rome?",
        metadata={"text": "test metadata"},
        document=Document(
            id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            content="My name is Giorgio and I live in Rome.",
            score=0.33144005810482535,
        ),
        probability=0.7661304473876953,
        start=11,
        end=24,
    )

    assert extracted_answer.to_dict(flatten=False) == {
        "data": "Giorgio and I",
        "query": "Who lives in Rome?",
        "metadata": {"text": "test metadata"},
        "document": {
            "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            "content": "My name is Giorgio and I live in Rome.",
            "dataframe": None,
            "blob": None,
            "meta": {},
            "score": 0.33144005810482535,
            "embedding": None,
        },
        "probability": 0.7661304473876953,
        "start": 11,
        "end": 24,
    }


def test_extracted_answer_from_dict():
    data = {
        "data": "Giorgio and I",
        "query": "Who lives in Rome?",
        "document": {
            "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            "content": "My name is Giorgio and I live in Rome.",
            "dataframe": None,
            "blob": None,
            "score": 0.33144005810482535,
            "embedding": None,
        },
        "probability": 0.7661304473876953,
        "start": 11,
        "end": 24,
        "text": "test metadata",
    }

    assert ExtractedAnswer.from_dict(data) == ExtractedAnswer(
        data="Giorgio and I",
        query="Who lives in Rome?",
        metadata={"text": "test metadata"},
        document=Document(
            id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            content="My name is Giorgio and I live in Rome.",
            score=0.33144005810482535,
        ),
        probability=0.7661304473876953,
        start=11,
        end=24,
    )


def test_extracted_answer_from_dict_without_flattening():
    data = {
        "data": "Giorgio and I",
        "query": "Who lives in Rome?",
        "metadata": {"text": "test metadata"},
        "document": {
            "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            "content": "My name is Giorgio and I live in Rome.",
            "dataframe": None,
            "blob": None,
            "meta": {},
            "score": 0.33144005810482535,
            "embedding": None,
        },
        "probability": 0.7661304473876953,
        "start": 11,
        "end": 24,
    }

    assert ExtractedAnswer.from_dict(data) == ExtractedAnswer(
        data="Giorgio and I",
        query="Who lives in Rome?",
        metadata={"text": "test metadata"},
        document=Document(
            id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
            content="My name is Giorgio and I live in Rome.",
            score=0.33144005810482535,
        ),
        probability=0.7661304473876953,
        start=11,
        end=24,
    )


def test_extracted_answer_from_dict_with_flat_and_non_flat_meta():
    with pytest.raises(ValueError, match="Pass either the 'metadata' parameter or flattened metadata keys"):
        ExtractedAnswer.from_dict(
            {
                "data": "Giorgio and I",
                "query": "Who lives in Rome?",
                "metadata": {"text": "test metadata"},
                "document": {
                    "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                    "content": "My name is Giorgio and I live in Rome.",
                    "dataframe": None,
                    "blob": None,
                    "meta": {},
                    "score": 0.33144005810482535,
                    "embedding": None,
                },
                "probability": 0.7661304473876953,
                "start": 11,
                "end": 24,
                "text": "test metadata other",
            }
        )


def test_generated_answer_to_dict():
    generated_answer = GeneratedAnswer(
        data="Mark and I",
        query="Who lives in Berlin?",
        metadata={"text": "test metadata"},
        documents=[
            Document(
                id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                content="My name is Mark and I live in Berlin.",
                score=0.33144005810482535,
            ),
            Document(
                id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                content="My name is Giorgio and I live in Rome.",
                score=-0.17938556566116537,
            ),
            Document(
                id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                content="My name is Jean and I live in Paris.",
                score=-0.17938556566116537,
            ),
        ],
    )

    assert generated_answer.to_dict() == {
        "data": "Mark and I",
        "query": "Who lives in Berlin?",
        "documents": [
            {
                "id": "10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                "content": "My name is Mark and I live in Berlin.",
                "dataframe": None,
                "blob": None,
                "score": 0.33144005810482535,
                "embedding": None,
            },
            {
                "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                "content": "My name is Giorgio and I live in Rome.",
                "dataframe": None,
                "blob": None,
                "score": -0.17938556566116537,
                "embedding": None,
            },
            {
                "id": "6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                "content": "My name is Jean and I live in Paris.",
                "dataframe": None,
                "blob": None,
                "score": -0.17938556566116537,
                "embedding": None,
            },
        ],
        "text": "test metadata",
    }


def test_generated_answer_to_dict_without_flattening():
    generated_answer = GeneratedAnswer(
        data="Mark and I",
        query="Who lives in Berlin?",
        metadata={"text": "test metadata"},
        documents=[
            Document(
                id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                content="My name is Mark and I live in Berlin.",
                score=0.33144005810482535,
            ),
            Document(
                id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                content="My name is Giorgio and I live in Rome.",
                score=-0.17938556566116537,
            ),
            Document(
                id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                content="My name is Jean and I live in Paris.",
                score=-0.17938556566116537,
            ),
        ],
    )

    assert generated_answer.to_dict(flatten=False) == {
        "data": "Mark and I",
        "query": "Who lives in Berlin?",
        "metadata": {"text": "test metadata"},
        "documents": [
            {
                "id": "10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                "content": "My name is Mark and I live in Berlin.",
                "dataframe": None,
                "blob": None,
                "meta": {},
                "score": 0.33144005810482535,
                "embedding": None,
            },
            {
                "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                "content": "My name is Giorgio and I live in Rome.",
                "dataframe": None,
                "blob": None,
                "meta": {},
                "score": -0.17938556566116537,
                "embedding": None,
            },
            {
                "id": "6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                "content": "My name is Jean and I live in Paris.",
                "dataframe": None,
                "blob": None,
                "meta": {},
                "score": -0.17938556566116537,
                "embedding": None,
            },
        ],
    }


def test_generated_answer_from_dict():
    data = {
        "data": "Mark and I",
        "query": "Who lives in Berlin?",
        "documents": [
            {
                "id": "10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                "content": "My name is Mark and I live in Berlin.",
                "dataframe": None,
                "blob": None,
                "score": 0.33144005810482535,
                "embedding": None,
            },
            {
                "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                "content": "My name is Giorgio and I live in Rome.",
                "dataframe": None,
                "blob": None,
                "score": -0.17938556566116537,
                "embedding": None,
            },
            {
                "id": "6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                "content": "My name is Jean and I live in Paris.",
                "dataframe": None,
                "blob": None,
                "score": -0.17938556566116537,
                "embedding": None,
            },
        ],
        "text": "test metadata",
    }

    assert GeneratedAnswer.from_dict(data) == GeneratedAnswer(
        data="Mark and I",
        query="Who lives in Berlin?",
        metadata={"text": "test metadata"},
        documents=[
            Document(
                id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                content="My name is Mark and I live in Berlin.",
                score=0.33144005810482535,
            ),
            Document(
                id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                content="My name is Giorgio and I live in Rome.",
                score=-0.17938556566116537,
            ),
            Document(
                id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                content="My name is Jean and I live in Paris.",
                score=-0.17938556566116537,
            ),
        ],
    )


def test_generated_answer_from_dict_without_flattening():
    data = {
        "data": "Mark and I",
        "query": "Who lives in Berlin?",
        "metadata": {"text": "test metadata"},
        "documents": [
            {
                "id": "10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                "content": "My name is Mark and I live in Berlin.",
                "dataframe": None,
                "blob": None,
                "meta": {},
                "score": 0.33144005810482535,
                "embedding": None,
            },
            {
                "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                "content": "My name is Giorgio and I live in Rome.",
                "dataframe": None,
                "blob": None,
                "meta": {},
                "score": -0.17938556566116537,
                "embedding": None,
            },
            {
                "id": "6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                "content": "My name is Jean and I live in Paris.",
                "dataframe": None,
                "blob": None,
                "meta": {},
                "score": -0.17938556566116537,
                "embedding": None,
            },
        ],
    }

    assert GeneratedAnswer.from_dict(data) == GeneratedAnswer(
        data="Mark and I",
        query="Who lives in Berlin?",
        metadata={"text": "test metadata"},
        documents=[
            Document(
                id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                content="My name is Mark and I live in Berlin.",
                score=0.33144005810482535,
            ),
            Document(
                id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                content="My name is Giorgio and I live in Rome.",
                score=-0.17938556566116537,
            ),
            Document(
                id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                content="My name is Jean and I live in Paris.",
                score=-0.17938556566116537,
            ),
        ],
    )


def test_generated_answer_from_dict_with_flat_and_non_flat_meta():
    with pytest.raises(ValueError, match="Pass either the 'metadata' parameter or flattened metadata keys"):
        GeneratedAnswer.from_dict(
            {
                "data": "Mark and I",
                "query": "Who lives in Berlin?",
                "metadata": {"text": "test metadata"},
                "documents": [
                    {
                        "id": "10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                        "content": "My name is Mark and I live in Berlin.",
                        "dataframe": None,
                        "blob": None,
                        "meta": {},
                        "score": 0.33144005810482535,
                        "embedding": None,
                    },
                    {
                        "id": "fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                        "content": "My name is Giorgio and I live in Rome.",
                        "dataframe": None,
                        "blob": None,
                        "meta": {},
                        "score": -0.17938556566116537,
                        "embedding": None,
                    },
                    {
                        "id": "6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                        "content": "My name is Jean and I live in Paris.",
                        "dataframe": None,
                        "blob": None,
                        "meta": {},
                        "score": -0.17938556566116537,
                        "embedding": None,
                    },
                ],
                "text": "test metadata other",
            }
        )
