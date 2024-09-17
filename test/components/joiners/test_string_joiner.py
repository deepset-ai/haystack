# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Pipeline
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.components.joiners.string_joiner import StringJoiner
from haystack.components.builders.prompt_builder import PromptBuilder


class TestAnswerJoiner:
    def test_init(self):
        joiner = StringJoiner()
        assert isinstance(joiner, StringJoiner)

    def test_to_dict(self):
        joiner = StringJoiner()
        data = component_to_dict(joiner)
        assert data == {"type": "haystack.components.joiners.string_joiner.StringJoiner", "init_parameters": {}}

    def test_from_dict(self):
        data = {"type": "haystack.components.joiners.string_joiner.StringJoiner", "init_parameters": {}}
        string_joiner = component_from_dict(StringJoiner, data=data, name="string_joiner")
        assert isinstance(string_joiner, StringJoiner)

    def test_empty_list(self):
        joiner = StringJoiner()
        result = joiner.run([])
        assert result == {"strings": []}

    def test_single_string(self):
        joiner = StringJoiner()
        result = joiner.run("a")
        assert result == {"strings": ["a"]}

    def test_two_strings(self):
        joiner = StringJoiner()
        result = joiner.run(["a", "b"])
        assert result == {"strings": ["a", "b"]}

    @pytest.mark.integration
    def test_with_pipeline(self):
        string_1 = "What's Natural Language Processing?"
        string_2 = "What's is life?"

        pipe = Pipeline()
        pipe.add_component("prompt_builder_1", PromptBuilder("Builder 1: {{query}}"))
        pipe.add_component("prompt_builder_2", PromptBuilder("Builder 2: {{query}}"))
        pipe.add_component("joiner", StringJoiner())

        pipe.connect("prompt_builder_1.prompt", "joiner.strings")
        pipe.connect("prompt_builder_2.prompt", "joiner.strings")

        results = pipe.run(data={"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}})

        assert "joiner" in results
        assert len(results["joiner"]["strings"]) == 2
        assert results["joiner"]["strings"] == [
            "Builder 1: What's Natural Language Processing?",
            "Builder 2: What's is life?",
        ]
