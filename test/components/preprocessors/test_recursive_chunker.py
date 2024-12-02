import pytest
from haystack.components.preprocessors.recursive_chunker import RecursiveChunker


def test_init_with_negative_overlap():
    with pytest.raises(ValueError):
        _ = RecursiveChunker(chunk_size=20, chunk_overlap=-1, separators=["."])


def test_init_with_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        _ = RecursiveChunker(chunk_size=10, chunk_overlap=15, separators=["."])


def test_apply_overlap_no_overlap():
    # Test the case where there is no overlap between chunks
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = chunker._apply_overlap(chunks)
    assert result == ["chunk1", "chunk2", "chunk3"]


def test_apply_overlap_with_overlap():
    # Test the case where there is overlap between chunks
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=4, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = chunker._apply_overlap(chunks)
    assert result == ["chunk1", "unk1chunk2", "unk2chunk3"]


def test_apply_overlap_single_chunk():
    # Test the case where there is only one chunk
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=3, separators=["."])
    chunks = ["chunk1"]
    result = chunker._apply_overlap(chunks)
    assert result == ["chunk1"]


def test_chunk_text_smaller_than_chunk_size():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0, separators=["."])
    text = "small text"
    chunks = chunker._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


@pytest.mark.parametrize("keep_separator", [True, False])
def test_chunk_text_with_simple_separator(keep_separator):
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0, separators=["."], keep_separator=keep_separator)

    text = "This is a test. Another sentence. And one more."
    chunks = chunker._chunk_text(text)

    if keep_separator:
        assert len(chunks) == 3
        assert chunks[0] == "This is a test."
        assert chunks[1] == " Another sentence."
        assert chunks[2] == " And one more."
    else:
        assert len(chunks) == 3
        assert chunks[0] == "This is a test"
        assert chunks[1] == " Another sentence"
        assert chunks[2] == " And one more"


def test_chunk_text_with_multiple_separators_recursive():
    chunker = RecursiveChunker(
        chunk_size=260, chunk_overlap=0, separators=["\n\n", "\n", ".", " "], keep_separator=True
    )
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games."""  # noqa: E501
    chunks = chunker._chunk_text(text)
    assert len(chunks) == 4
    assert chunks[0] == "Artificial intelligence (AI) - Introduction\n\n"
    assert (
        chunks[1]
        == "AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n"
    )  # noqa: E501
    assert chunks[2] == "AI technology is widely used throughout industry, government, and science."
    assert (
        chunks[3]
        == " Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games."
    )  # noqa: E501


def test_chunk_text_using_nltk_sentence():
    """
    This test includes abbreviations that are not handled by the simple sentence tokenizer based on "." and
    requires a more sophisticated sentence tokenizer like the one provided by NLTK.
    """

    chunker = RecursiveChunker(
        chunk_size=400, chunk_overlap=0, separators=["\n\n", "\n", "sentence", " "], keep_separator=True
    )
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."""  # noqa: E501

    chunks = chunker._chunk_text(text)
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
