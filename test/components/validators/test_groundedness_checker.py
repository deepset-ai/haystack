# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from haystack import Document
from haystack.components.validators.groundedness_checker import GroundednessChecker
from haystack.dataclasses import ChatMessage


class TestGroundednessChecker:
    def test_init_defaults(self):
        checker = GroundednessChecker()
        assert checker.max_claims == 10
        assert checker.trust_threshold == 0.5
        assert checker.block_contradicted is False
        assert checker.raise_on_failure is True

    def test_init_custom(self):
        checker = GroundednessChecker(max_claims=5, trust_threshold=0.8, block_contradicted=True)
        assert checker.max_claims == 5
        assert checker.trust_threshold == 0.8
        assert checker.block_contradicted is True

    def test_max_claims_clamped(self):
        assert GroundednessChecker(max_claims=100).max_claims == 20
        assert GroundednessChecker(max_claims=0).max_claims == 1

    def test_no_documents_returns_no_context(self):
        checker = GroundednessChecker()
        replies = [ChatMessage.from_assistant("Some reply text here for testing purposes.")]
        result = checker.run(replies=replies, documents=None)
        assert result["verdict"] == "no_context"
        assert result["trust_score"] == 0.0
        assert result["is_trusted"] is False
        assert result["verified_replies"] == ["Some reply text here for testing purposes."]

    def test_empty_documents_returns_no_context(self):
        checker = GroundednessChecker()
        replies = [ChatMessage.from_assistant("Some reply text here for testing purposes.")]
        result = checker.run(replies=replies, documents=[Document(content="")])
        assert result["verdict"] == "no_context"

    def test_empty_reply_passes_through(self):
        checker = GroundednessChecker()
        replies = [ChatMessage.from_assistant("")]
        docs = [Document(content="Some context about revenue and growth metrics.")]
        result = checker.run(replies=replies, documents=docs)
        assert result["verified_replies"] == [""]
        assert result["verdict"] == "no_claims"
        assert result["trust_score"] == 0.0

    def test_to_dict(self):
        checker = GroundednessChecker(
            max_claims=5, trust_threshold=0.7, block_contradicted=True, raise_on_failure=False
        )
        data = checker.to_dict()
        assert data["init_parameters"]["max_claims"] == 5
        assert data["init_parameters"]["trust_threshold"] == 0.7
        assert data["init_parameters"]["block_contradicted"] is True
        assert data["init_parameters"]["raise_on_failure"] is False
        assert "chat_generator" in data["init_parameters"]

    def test_from_dict(self):
        checker = GroundednessChecker(max_claims=5, trust_threshold=0.8, block_contradicted=True)
        data = checker.to_dict()
        restored = GroundednessChecker.from_dict(data)
        assert restored.max_claims == 5
        assert restored.trust_threshold == 0.8
        assert restored.block_contradicted is True

    def test_warm_up_delegates(self):
        mock_gen = MagicMock()
        mock_gen.warm_up = MagicMock()
        checker = GroundednessChecker(chat_generator=mock_gen)
        checker.warm_up()
        mock_gen.warm_up.assert_called_once()

    def test_warm_up_idempotent(self):
        mock_gen = MagicMock()
        mock_gen.warm_up = MagicMock()
        checker = GroundednessChecker(chat_generator=mock_gen)
        checker.warm_up()
        checker.warm_up()  # second call should be a no-op
        mock_gen.warm_up.assert_called_once()

    def test_warm_up_no_op_if_missing(self):
        mock_gen = MagicMock(spec=[])  # no warm_up attribute
        checker = GroundednessChecker(chat_generator=mock_gen)
        checker.warm_up()  # should not raise

    def test_run_all_supported(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant('["Revenue was $2.1B", "Growth was 12% YoY"]')]},
                {"replies": [ChatMessage.from_assistant(
                    '[{"claim": "Revenue was $2.1B", "verdict": "supported", "explanation": "Confirmed", "correction": null},'
                    '{"claim": "Growth was 12% YoY", "verdict": "supported", "explanation": "Confirmed", "correction": null}]'
                )]},
            ]
        )

        checker = GroundednessChecker(chat_generator=mock_gen)
        result = checker.run(
            replies=[ChatMessage.from_assistant("Revenue was $2.1B and growth was 12% YoY.")],
            documents=[Document(content="Q3 Earnings: Revenue was $2.1B, representing 12% YoY growth.")],
        )
        assert result["trust_score"] == 1.0
        assert result["verdict"] == "all_supported"
        assert result["is_trusted"] is True
        assert len(result["claims"]) == 2

    def test_run_with_contradictions(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant('["Revenue was $2.4B"]')]},
                {"replies": [ChatMessage.from_assistant(
                    '[{"claim": "Revenue was $2.4B", "verdict": "contradicted", '
                    '"explanation": "Context says $2.1B", "correction": "Revenue was $2.1B"}]'
                )]},
            ]
        )

        checker = GroundednessChecker(chat_generator=mock_gen)
        result = checker.run(
            replies=[ChatMessage.from_assistant("Revenue was $2.4B in Q3.")],
            documents=[Document(content="Q3 Earnings: Revenue was $2.1B.")],
        )
        assert result["trust_score"] == 0.0
        assert result["verdict"] == "has_contradictions"
        assert result["is_trusted"] is False
        assert result["claims"][0]["correction"] == "Revenue was $2.1B"

    def test_block_contradicted_replaces_text(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant('["Revenue was $2.4B"]')]},
                {"replies": [ChatMessage.from_assistant(
                    '[{"claim": "Revenue was $2.4B", "verdict": "contradicted", '
                    '"explanation": "Wrong", "correction": "Revenue was $2.1B"}]'
                )]},
            ]
        )

        checker = GroundednessChecker(chat_generator=mock_gen, block_contradicted=True)
        result = checker.run(
            replies=[ChatMessage.from_assistant("Revenue was $2.4B in Q3.")],
            documents=[Document(content="Q3 Earnings: Revenue was $2.1B.")],
        )
        assert "[CORRECTED: Revenue was $2.1B]" in result["verified_replies"][0]

    def test_multiple_replies(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant('["Claim A"]')]},
                {"replies": [ChatMessage.from_assistant(
                    '[{"claim": "Claim A", "verdict": "supported", "explanation": "OK", "correction": null}]'
                )]},
                {"replies": [ChatMessage.from_assistant('["Claim B"]')]},
                {"replies": [ChatMessage.from_assistant(
                    '[{"claim": "Claim B", "verdict": "contradicted", "explanation": "Wrong", "correction": "Fixed B"}]'
                )]},
            ]
        )

        checker = GroundednessChecker(chat_generator=mock_gen)
        result = checker.run(
            replies=[
                ChatMessage.from_assistant("Claim A is stated here in this reply."),
                ChatMessage.from_assistant("Claim B is stated here in this reply."),
            ],
            documents=[Document(content="The context contains Claim A but contradicts Claim B.")],
        )
        assert len(result["verified_replies"]) == 2
        assert result["trust_score"] == 0.5
        assert result["verdict"] == "has_contradictions"

    def test_raise_on_failure_true(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(side_effect=RuntimeError("LLM failed"))

        checker = GroundednessChecker(chat_generator=mock_gen, raise_on_failure=True)
        with pytest.raises(RuntimeError, match="LLM failed"):
            checker.run(
                replies=[ChatMessage.from_assistant("Revenue was $2.1B for the quarter.")],
                documents=[Document(content="Revenue data here.")],
            )

    def test_raise_on_failure_false(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(side_effect=RuntimeError("LLM failed"))

        checker = GroundednessChecker(chat_generator=mock_gen, raise_on_failure=False)
        result = checker.run(
            replies=[ChatMessage.from_assistant("Revenue was $2.1B for the quarter.")],
            documents=[Document(content="Revenue data here.")],
        )
        assert result["verified_replies"] == ["Revenue was $2.1B for the quarter."]
        assert result["verdict"] == "no_claims"

    def test_malformed_json_from_llm(self):
        mock_gen = MagicMock()
        mock_gen.run = MagicMock(
            return_value={"replies": [ChatMessage.from_assistant("This is not valid JSON at all")]}
        )

        checker = GroundednessChecker(chat_generator=mock_gen, raise_on_failure=False)
        result = checker.run(
            replies=[ChatMessage.from_assistant("Revenue was $2.1B for the quarter.")],
            documents=[Document(content="Revenue data here.")],
        )
        # Should gracefully handle and return no claims
        assert result["verdict"] == "no_claims"
