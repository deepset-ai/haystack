# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from pandas import DataFrame
from haystack import Document
from haystack.dataclasses import ByteStream, SparseEmbedding
from haystack.components.preprocessors import DocumentCleaner


class TestDocumentCleaner:
    def test_init(self):
        cleaner = DocumentCleaner()
        assert cleaner.remove_empty_lines is True
        assert cleaner.remove_extra_whitespaces is True
        assert cleaner.remove_repeated_substrings is False
        assert cleaner.remove_substrings is None
        assert cleaner.remove_regex is None
        assert cleaner.keep_id is False

    def test_non_text_document(self, caplog):
        with caplog.at_level(logging.WARNING):
            cleaner = DocumentCleaner()
            cleaner.run(documents=[Document()])
            assert "DocumentCleaner only cleans text documents but document.content for document ID" in caplog.text

    def test_single_document(self):
        with pytest.raises(TypeError, match="DocumentCleaner expects a List of Documents as input."):
            cleaner = DocumentCleaner()
            cleaner.run(documents=Document())

    def test_empty_list(self):
        cleaner = DocumentCleaner()
        result = cleaner.run(documents=[])
        assert result == {"documents": []}

    def test_remove_empty_lines(self):
        cleaner = DocumentCleaner(remove_extra_whitespaces=False)
        result = cleaner.run(
            documents=[
                Document(
                    content="This is a text with some words. \f"
                    ""
                    "There is a second sentence. "
                    ""
                    "And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert (
            result["documents"][0].content
            == "This is a text with some words. \fThere is a second sentence. And there is a third sentence."
        )

    def test_remove_whitespaces(self):
        cleaner = DocumentCleaner(remove_empty_lines=False)
        result = cleaner.run(
            documents=[
                Document(
                    content=" This is a text with some words. "
                    ""
                    "There is a second sentence.  "
                    ""
                    "And there  is a third sentence.\f "
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == (
            "This is a text with some words. " "" "There is a second sentence. " "" "And there is a third sentence.\f"
        )

    def test_remove_substrings(self):
        cleaner = DocumentCleaner(remove_substrings=["This", "A", "words", "🪲"])
        result = cleaner.run(documents=[Document(content="This is a text with some words.\f🪲")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == " is a text with some .\f"

    def test_remove_regex(self):
        cleaner = DocumentCleaner(remove_regex=r"\s\s+")
        result = cleaner.run(documents=[Document(content="This is a  text \f with   some words.")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "This is a text\fwith some words."

    def test_remove_repeated_substrings(self):
        cleaner = DocumentCleaner(
            remove_empty_lines=False, remove_extra_whitespaces=False, remove_repeated_substrings=True
        )

        text = """First Page\fThis is a header.
        Page  of
        2
        4
        Lorem ipsum dolor sit amet
        This is a footer number 1
        This is footer number 2This is a header.
        Page  of
        3
        4
        Sid ut perspiciatis unde
        This is a footer number 1
        This is footer number 2This is a header.
        Page  of
        4
        4
        Sed do eiusmod tempor.
        This is a footer number 1
        This is footer number 2"""

        expected_text = """First Page\f 2
        4
        Lorem ipsum dolor sit amet 3
        4
        Sid ut perspiciatis unde 4
        4
        Sed do eiusmod tempor."""
        result = cleaner.run(documents=[Document(content=text)])
        assert result["documents"][0].content == expected_text

    def test_copy_metadata(self):
        cleaner = DocumentCleaner()
        documents = [
            Document(content="Text. ", meta={"name": "doc 0"}),
            Document(content="Text. ", meta={"name": "doc 1"}),
        ]
        result = cleaner.run(documents=documents)
        assert len(result["documents"]) == 2
        assert result["documents"][0].id != result["documents"][1].id
        for doc, cleaned_doc in zip(documents, result["documents"]):
            assert doc.meta == cleaned_doc.meta
            assert cleaned_doc.content == "Text."

    def test_keep_id_does_not_alter_document_ids(self):
        cleaner = DocumentCleaner(keep_id=True)
        documents = [Document(content="Text. ", id="1"), Document(content="Text. ", id="2")]
        result = cleaner.run(documents=documents)
        assert len(result["documents"]) == 2
        assert result["documents"][0].id == "1"
        assert result["documents"][1].id == "2"

    def test_unicode_normalization(self):
        text = """\
        ｱｲｳｴｵ
        Comment ça va
        مرحبا بالعالم
        em Space"""

        expected_text_NFC = """\
        ｱｲｳｴｵ
        Comment ça va
        مرحبا بالعالم
        em Space"""

        expected_text_NFD = """\
        ｱｲｳｴｵ
        Comment ça va
        مرحبا بالعالم
        em Space"""

        expected_text_NFKC = """\
        アイウエオ
        Comment ça va
        مرحبا بالعالم
        em Space"""

        expected_text_NFKD = """\
        アイウエオ
        Comment ça va
        مرحبا بالعالم
        em Space"""

        nfc_cleaner = DocumentCleaner(unicode_normalization="NFC", remove_extra_whitespaces=False)
        nfd_cleaner = DocumentCleaner(unicode_normalization="NFD", remove_extra_whitespaces=False)
        nfkc_cleaner = DocumentCleaner(unicode_normalization="NFKC", remove_extra_whitespaces=False)
        nfkd_cleaner = DocumentCleaner(unicode_normalization="NFKD", remove_extra_whitespaces=False)

        nfc_result = nfc_cleaner.run(documents=[Document(content=text)])
        nfd_result = nfd_cleaner.run(documents=[Document(content=text)])
        nfkc_result = nfkc_cleaner.run(documents=[Document(content=text)])
        nfkd_result = nfkd_cleaner.run(documents=[Document(content=text)])

        assert nfc_result["documents"][0].content == expected_text_NFC
        assert nfd_result["documents"][0].content == expected_text_NFD
        assert nfkc_result["documents"][0].content == expected_text_NFKC
        assert nfkd_result["documents"][0].content == expected_text_NFKD

    def test_ascii_only(self):
        text = """\
        ｱｲｳｴｵ
        Comment ça va
        Á
        مرحبا بالعالم
        em Space"""

        expected_text = """\
        \n\
        Comment ca va
        A
         \n\
        em Space"""

        cleaner = DocumentCleaner(ascii_only=True, remove_extra_whitespaces=False, remove_empty_lines=False)
        result = cleaner.run(documents=[Document(content=text)])
        assert result["documents"][0].content == expected_text

    def test_other_document_fields_are_not_lost(self):
        cleaner = DocumentCleaner(keep_id=True)
        document = Document(
            content="This is a text with some words. \n"
            ""
            "There is a second sentence. \n"
            ""
            "And there is a third sentence.\n",
            dataframe=DataFrame({"col1": [1], "col2": [2]}),
            blob=ByteStream.from_string("some_data"),
            meta={"data": 1},
            score=0.1,
            embedding=[0.1, 0.2, 0.3],
            sparse_embedding=SparseEmbedding([0, 2], [0.1, 0.3]),
        )
        res = cleaner.run(documents=[document])

        assert len(res) == 1
        assert len(res["documents"])
        assert res["documents"][0].id == document.id
        assert res["documents"][0].content == (
            "This is a text with some words. There is a second sentence. And there is a third sentence."
        )
        assert res["documents"][0].dataframe.equals(document.dataframe)
        assert res["documents"][0].blob == document.blob
        assert res["documents"][0].meta == document.meta
        assert res["documents"][0].score == document.score
        assert res["documents"][0].embedding == document.embedding
        assert res["documents"][0].sparse_embedding == document.sparse_embedding
