# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Document
from haystack.components.rankers.hugging_face_api import HuggingFaceAPIRanker


class TestHuggingFaceAPIRanker:
    def test_to_dict(self):
        component = HuggingFaceAPIRanker(
            url="https://api.my-tei-service.com", top_k=5, timeout=30, token="my_api_token"
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.hugging_face_api.HuggingFaceAPIRanker",
            "init_parameters": {
                "url": "https://api.my-tei-service.com",
                "top_k": 5,
                "timeout": 30,
                "token": "my_api_token",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = HuggingFaceAPIRanker(
            url="https://api.my-tei-service.com", top_k=5, timeout=30, token="my_api_token"
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.hugging_face_api.HuggingFaceAPIRanker",
            "init_parameters": {
                "url": "https://api.my-tei-service.com",
                "top_k": 5,
                "timeout": 30,
                "token": "my_api_token",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.rankers.hugging_face_api.HuggingFaceAPIRanker",
            "init_parameters": {
                "url": "https://api.my-tei-service.com",
                "top_k": 5,
                "timeout": 30,
                "token": "my_api_token",
            },
        }

        component = HuggingFaceAPIRanker.from_dict(data)
        assert component.url == "https://api.my-tei-service.com"
        assert component.top_k == 5
        assert component.timeout == 30
        assert component.token == "my_api_token"

    @pytest.mark.integration
    def test_run(self):
        """
        Test if the component ranks documents correctly.
        """
        ranker = HuggingFaceAPIRanker(url="http://localhost:3000", timeout=5)

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"
        expected_scores = [0.07423137, 0.89478946, 0.96765566]

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted(expected_scores, reverse=True)
        assert docs_after[0].score == pytest.approx(sorted_scores[0], abs=1e-6)
        assert docs_after[1].score == pytest.approx(sorted_scores[1], abs=1e-6)
        assert docs_after[2].score == pytest.approx(sorted_scores[2], abs=1e-6)

    @pytest.mark.integration
    def test_run_top_k(self):
        """
        Test if the component ranks documents correctly with a custom top_k.
        """
        ranker = HuggingFaceAPIRanker(url="http://localhost:3000", timeout=5, top_k=2)

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 2
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores

    @pytest.mark.integration
    def test_run_single_document(self):
        """
        Test if the component runs with a single document.
        """
        ranker = HuggingFaceAPIRanker(url="http://localhost:3000", timeout=5)
        docs_before = [Document(content="Berlin")]
        output = ranker.run(query="City in Germany", documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 1

    @pytest.mark.asyncio
    async def test_run_async(self):
        """
        Test if the component ranks documents correctly.
        """
        ranker = HuggingFaceAPIRanker(url="http://localhost:3000", timeout=5)

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"
        expected_scores = [0.07423137, 0.89478946, 0.96765566]

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = await ranker.run_async(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted(expected_scores, reverse=True)
        assert docs_after[0].score == pytest.approx(sorted_scores[0], abs=1e-6)
        assert docs_after[1].score == pytest.approx(sorted_scores[1], abs=1e-6)
        assert docs_after[2].score == pytest.approx(sorted_scores[2], abs=1e-6)
