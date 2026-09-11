# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.preprocessors import PIIScrubber


class TestPIIScrubber:
    def test_scrubs_email(self):
        scrubber = PIIScrubber()
        result = scrubber.run(documents=[Document(content="Contact jane@example.com today")])
        doc = result["documents"][0]
        assert doc.content is not None
        assert "jane@example.com" not in doc.content
        assert "[EMAIL_1]" in doc.content

    def test_scrubs_phone(self):
        scrubber = PIIScrubber(entities=["phone"])
        result = scrubber.run(documents=[Document(content="Call 415-555-1234 now")])
        doc = result["documents"][0]
        assert doc.content is not None
        assert "415-555-1234" not in doc.content

    def test_credit_card_luhn(self):
        scrubber = PIIScrubber(entities=["credit_card"])
        valid = scrubber.run(documents=[Document(content="Card 4242424242424242")])["documents"][0]
        assert valid.content is not None
        assert "4242424242424242" not in valid.content
        invalid = scrubber.run(documents=[Document(content="Num 1234567890123")])["documents"][0]
        assert invalid.content is not None
        assert "1234567890123" in invalid.content

    def test_entities_opt_in(self):
        scrubber = PIIScrubber(entities=["email"])
        result = scrubber.run(documents=[Document(content="jane@example.com 415-555-1234")])["documents"][0]
        assert result.content is not None
        assert "415-555-1234" in result.content

    def test_audit_meta(self):
        scrubber = PIIScrubber()
        doc = scrubber.run(documents=[Document(content="jane@example.com")])["documents"][0]
        assert doc.meta["pii_redactions"][0]["entity"] == "email"
        assert doc.meta["pii_redactions"][0]["count"] == 1

    def test_no_pii_no_meta(self):
        doc = PIIScrubber().run(documents=[Document(content="Hello world")])["documents"][0]
        assert "pii_redactions" not in doc.meta
        assert doc.content == "Hello world"

    def test_none_content_passthrough(self):
        doc = Document(content=None)
        result = PIIScrubber().run(documents=[doc])["documents"][0]
        assert result.content is None
        assert "pii_redactions" not in result.meta

    def test_invalid_input(self):
        with pytest.raises(TypeError, match="PIIScrubber expects a List of Documents"):
            PIIScrubber().run(documents="not a list")  # type: ignore[arg-type]

    def test_invalid_entity(self):
        with pytest.raises(ValueError, match="Unsupported entity"):
            PIIScrubber(entities=["passport"])  # type: ignore[list-item]

    def test_invalid_template(self):
        with pytest.raises(ValueError, match="replacement_template"):
            PIIScrubber(replacement_template="[PII]")

    def test_to_dict(self):
        data = PIIScrubber().to_dict()
        assert data["type"] == "haystack.components.preprocessors.pii_scrubber.PIIScrubber"

    def test_from_dict(self):
        scrubber = PIIScrubber.from_dict(
            {
                "type": "haystack.components.preprocessors.pii_scrubber.PIIScrubber",
                "init_parameters": {
                    "entities": ["email"],
                    "replacement_template": "[{entity}_{index}]",
                    "keep_id": True,
                },
            }
        )
        assert scrubber.entities == ["email"]
        assert scrubber.keep_id is True

    def test_roundtrip_serialization(self):
        scrubber = PIIScrubber(entities=["email"], keep_id=True)
        data = scrubber.to_dict()
        restored = PIIScrubber.from_dict(data)
        result = restored.run(documents=[Document(content="Hi jane@example.com")])
        assert result["documents"][0].content is not None
        assert "[EMAIL_1]" in result["documents"][0].content
