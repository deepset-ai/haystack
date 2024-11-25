import pytest
from haystack.components.preprocessors.recursive_chunker import RecursiveChunker
from haystack.dataclasses import Document


@pytest.mark.parametrize("keep_separator", [True, False])
def test_chunk_text_with_simple_separator(chunk_size, chunk_overlap, separators, keep_separator):
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0, separators=["."], keep_separator=keep_separator)

    text = "This is a test. Another sentence. And one more."
    chunks = chunker._chunk_text(text)

    assert len(chunks) == 3
    assert chunks[0] == "This is a test."
    assert chunks[1] == ". Another sentence."
    assert chunks[2] == ". And one more."


def test_chunk_text_with_multiple_separators_recursive():
    # try: paragraph, newline, sentence, space

    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0, separators=["\n\n", "\n", ".", " "], keep_separator=True)

    # This text has paragraph breaks, newlines, sentences, and spaces
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. Such machines may be called AIs.

AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go).[2] However, many AI applications are not perceived as AI: "A lot of cutting edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."
"""

    chunks = chunker._chunk_text(text)
