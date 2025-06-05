# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document
from haystack.components.preprocessors.chinese_document_splitter import ChineseDocumentSplitter


class TestChineseDocumentSplitter:
    @pytest.fixture
    def sample_text(self) -> str:
        return "这是第一句话，也是故事的开端，紧接着是第二句话，渐渐引出了背景；随后，翻开新/f的一页，我们读到了这一页的第一句话，继续延展出情节的发展，直到这页的第二句话将整段文字温柔地收束于平静之中。"

    def test_split_by_word(self, sample_text):
        """
        Test splitting by word.

        Note on Chinese words:
        Unlike English where words are usually separated by spaces,
        Chinese text is written continuously without spaces between words.
        Chinese words can consist of multiple characters.
        For example, the English word "America" is translated to "美国" in Chinese,
        which consists of two characters but is treated as a single word.
        Similarly, "Portugal" is "葡萄牙" in Chinese,
        three characters but one word.
        Therefore, splitting by word means splitting by these multi-character tokens,
        not simply by single characters or spaces.
        """
        splitter = ChineseDocumentSplitter(
            split_by="word", language="zh", particle_size="coarse", split_length=5, split_overlap=0
        )
        if hasattr(splitter, "warm_up"):
            splitter.warm_up()

        result = splitter.run(documents=[Document(content=sample_text)])
        docs = result["documents"]

        assert all(isinstance(doc, Document) for doc in docs)
        assert all(len(doc.content.strip()) <= 10 for doc in docs)

    def test_split_by_sentence(self, sample_text):
        splitter = ChineseDocumentSplitter(
            split_by="sentence", language="zh", particle_size="coarse", split_length=10, split_overlap=0
        )
        if hasattr(splitter, "warm_up"):
            splitter.warm_up()

        result = splitter.run(documents=[Document(content=sample_text)])
        docs = result["documents"]

        assert all(isinstance(doc, Document) for doc in docs)
        assert all(doc.content.strip() != "" for doc in docs)
        assert any("。" in doc.content for doc in docs), "Expected at least one chunk containing a full stop."

    def test_respect_sentence_boundary(self):
        """Test that respect_sentence_boundary=True avoids splitting sentences"""
        text = "这是第一句话，这是第二句话，这是第三句话。这是第四句话，这是第五句话，这是第六句话！这是第七句话，这是第八句话，这是第九句话？"
        doc = Document(content=text)

        splitter = ChineseDocumentSplitter(
            split_by="word", split_length=10, split_overlap=3, language="zh", respect_sentence_boundary=True
        )
        splitter.warm_up()
        result = splitter.run(documents=[doc])
        docs = result["documents"]

        print(f"Total chunks created: {len(docs)}.")
        for i, d in enumerate(docs):
            print(f"\nChunk {i + 1}:\n{d.content}")
            # Optional: check that sentences are not cut off
            assert d.content.strip().endswith(("。", "！", "？")), "Sentence was cut off!"

    def test_overlap_chunks_with_long_text(self):
        """Test split_overlap parameter to ensure there is clear overlap between chunks of long text"""
        text = (
            "月光轻轻洒落，林中传来阵阵狼嚎，夜色悄然笼罩一切。"
            "树叶在微风中沙沙作响，影子在地面上摇曳不定。"
            "一只猫头鹰静静地眨了眨眼，从枝头注视着四周……"
            "远处的小溪哗啦啦地流淌，仿佛在向石头倾诉着什么。"
            "“咔嚓”一声，某处的树枝突然断裂，然后恢复了寂静。"
            "空气中弥漫着松树与湿土的气息，令人心安。"
            "一只狐狸悄然出现，又迅速消失在灌木丛中。"
            "天上的星星闪烁着，仿佛在诉说古老的故事。"
            "时间仿佛停滞了……"
            "万物静候，聆听着夜的呼吸！"
        )
        doc = Document(content=text)

        splitter = ChineseDocumentSplitter(
            split_by="word", language="zh", split_length=30, split_overlap=10, particle_size="coarse"
        )
        if hasattr(splitter, "warm_up"):
            splitter.warm_up()

        result = splitter.run(documents=[doc])
        docs = result["documents"]

        print(f"Total chunks generated: {len(docs)}.")
        for i, d in enumerate(docs):
            print(f"\nChunk {i + 1}:\n{d.content}")

        assert len(docs) > 1, "Expected multiple chunks to be generated"

        max_len_allowed = 80  # Allow a somewhat relaxed max chunk length
        assert all(len(doc.content) <= max_len_allowed for doc in docs), (
            f"Some chunks exceed {max_len_allowed} characters"
        )

        def has_any_overlap(suffix: str, prefix: str) -> bool:
            """
            Check if suffix and prefix have at least one continuous overlapping character sequence.
            Tries from the longest possible overlap down to 1 character.
            Returns True if any overlap found.
            """
            max_check_len = min(len(suffix), len(prefix))
            for length in range(max_check_len, 0, -1):
                if suffix[-length:] == prefix[:length]:
                    return True
            return False

        for i in range(1, len(docs)):
            prev_chunk = docs[i - 1].content
            curr_chunk = docs[i].content

            # Take last 20 chars of prev chunk and first 20 chars of current chunk to check overlap
            overlap_prev = prev_chunk[-20:]
            overlap_curr = curr_chunk[:20]

            assert has_any_overlap(overlap_prev, overlap_curr), (
                f"Chunks {i} and {i + 1} do not overlap. "
                f"Tail (up to 20 chars): '{overlap_prev}' vs Head (up to 20 chars): '{overlap_curr}'"
            )
