import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors.recursive_splitter import RecursiveDocumentSplitter
from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter


def test_get_custom_sentence_tokenizer_success():
    tokenizer = RecursiveDocumentSplitter._get_custom_sentence_tokenizer()
    assert isinstance(tokenizer, SentenceSplitter)


def test_init_with_negative_overlap():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=20, split_overlap=-1, separators=["."])


def test_init_with_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=10, split_overlap=15, separators=["."])


def test_init_with_invalid_separators():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(separators=[".", 2])


def test_init_with_negative_split_length():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=-1, separators=["."])


def test_apply_overlap_no_overlap():
    # Test the case where there is no overlap between chunks
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "chunk2", "chunk3"]


def test_apply_overlap_with_overlap():
    # Test the case where there is overlap between chunks
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=4, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "unk1chunk2", "unk2chunk3"]


def test_apply_overlap_single_chunk():
    # Test the case where there is only one chunk
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=3, separators=["."])
    chunks = ["chunk1"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1"]


def test_chunk_text_smaller_than_chunk_size():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    text = "small text"
    chunks = splitter._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_split_by_period():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    text = "This is a test. Another sentence. And one more."
    chunks = splitter._chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "This is a test."
    assert chunks[1] == " Another sentence."
    assert chunks[2] == " And one more."


def test_chunk_text_using_nltk_sentence():
    """
    This test includes abbreviations that are not handled by the simple sentence tokenizer based on "." and
    requires a more sophisticated sentence tokenizer like the one provided by NLTK.
    """

    splitter = RecursiveDocumentSplitter(split_length=400, split_overlap=0, separators=["\n\n", "\n", "sentence", " "])
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."""  # noqa: E501

    chunks = splitter._chunk_text(text)
    assert len(chunks) == 4
    assert chunks[0] == "Artificial intelligence (AI) - Introduction\n\n"
    assert (
        chunks[1]
        == "AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n"
    )  # noqa: E501
    assert chunks[2] == "AI technology is widely used throughout industry, government, and science."  # noqa: E501
    assert (
        chunks[3]
        == "Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."
    )  # noqa: E501


def test_recursive_splitter_empty_documents():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    empty_doc = Document(content="")
    doc_chunks = splitter.run([empty_doc])
    doc_chunks = doc_chunks["documents"]
    assert len(doc_chunks) == 0


def test_recursive_chunker_with_multiple_separators_recursive():
    splitter = RecursiveDocumentSplitter(split_length=260, split_overlap=0, separators=["\n\n", "\n", ".", " "])
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games."""  # noqa: E501

    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 4
    assert (
        doc_chunks[0].meta["original_id"]
        == doc_chunks[1].meta["original_id"]
        == doc_chunks[2].meta["original_id"]
        == doc_chunks[3].meta["original_id"]
        == doc.id
    )
    assert doc_chunks[0].content == "Artificial intelligence (AI) - Introduction\n\n"
    assert (
        doc_chunks[1].content
        == "AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n"
    )
    assert doc_chunks[2].content == "AI technology is widely used throughout industry, government, and science."
    assert (
        doc_chunks[3].content
        == " Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games."
    )


def test_recursive_chunker_split_document_with_overlap():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=11, separators=[".", " "])
    text = """A simple sentence1. A bright sentence2. A clever sentence3. A joyful sentence4"""

    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 4
    assert (
        doc_chunks[0].meta["original_id"]
        == doc_chunks[1].meta["original_id"]
        == doc_chunks[2].meta["original_id"]
        == doc_chunks[3].meta["original_id"]
        == doc.id
    )

    assert doc_chunks[0].content == "A simple sentence1."
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == 0
    assert doc_chunks[0].meta["_split_overlap"] == [{"doc_id": doc_chunks[1].id, "range": (0, 11)}]

    assert doc_chunks[1].content == " sentence1. A bright sentence2."
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == 8
    assert doc_chunks[1].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[0].id, "range": (8, 19)},
        {"doc_id": doc_chunks[2].id, "range": (0, 11)},
    ]

    assert doc_chunks[2].content == " sentence2. A clever sentence3."
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == 28
    assert doc_chunks[2].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[1].id, "range": (20, 31)},
        {"doc_id": doc_chunks[3].id, "range": (0, 11)},
    ]

    assert doc_chunks[3].content == " sentence3. A joyful sentence4"
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == 48
    assert doc_chunks[3].meta["_split_overlap"] == [{"doc_id": doc_chunks[2].id, "range": (20, 31)}]


def test_recursive_splitter_generate_pages():
    splitter = RecursiveDocumentSplitter(split_length=18, split_overlap=0, separators=[" "])
    doc = Document(content="This is some text. \f This text is on another page. \f This is the last page.")
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]
    assert len(doc_chunks) == 7
    for doc in doc_chunks:
        if doc.meta["split_id"] in [0, 1, 2]:
            assert doc.meta["page_number"] == 1
        if doc.meta["split_id"] in [3, 4]:
            assert doc.meta["page_number"] == 2
        if doc.meta["split_id"] in [5, 6]:
            assert doc.meta["page_number"] == 3


def test_recursive_splitter_separator_exists_but_split_length_too_small_fall_back_to_character_chunking():
    splitter = RecursiveDocumentSplitter(separators=[" "], split_length=2)
    doc = Document(content="This is some text. This is some more text.")
    result = splitter.run(documents=[doc])
    assert len(result["documents"]) == 21
    for doc in result["documents"]:
        assert len(doc.content) == 2


def test_recursive_splitter_generate_empty_chunks():
    splitter = RecursiveDocumentSplitter(split_length=15, separators=["\n\n", "\n"])
    text = "This is a test.\n\n\nAnother test.\n\n\n\nFinal test."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert chunks[0].content == "This is a test."
    assert chunks[1].content == "\nAnother test."
    assert chunks[2].content == "Final test."


def test_recursive_splitter_fallback_to_character_chunking():
    text = "abczdefzghizjkl"
    separators = ["\n\n", "\n", "z"]
    splitter = RecursiveDocumentSplitter(split_length=2, separators=separators)
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    for chunk in chunks:
        assert len(chunk.content) <= 2


def test_recursive_splitter_serialization_in_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("chunker", RecursiveDocumentSplitter(split_length=20, split_overlap=5, separators=["."]))
    pipeline_dict = pipeline.dumps()
    new_pipeline = Pipeline.loads(pipeline_dict)
    assert pipeline_dict == new_pipeline.dumps()
