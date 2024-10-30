# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from haystack import component, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)


@component
class StringJoiner:
    """
    Component to join strings from different components to a list of strings.

    ### Usage example

    ```python
    from haystack.components.joiners import StringJoiner
    from haystack.components.builders import PromptBuilder
    from haystack.core.pipeline import Pipeline

    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    string_1 = "What's Natural Language Processing?"
    string_2 = "What is life?"

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder_1", PromptBuilder("Builder 1: {{query}}"))
    pipeline.add_component("prompt_builder_2", PromptBuilder("Builder 2: {{query}}"))
    pipeline.add_component("string_joiner", StringJoiner())

    pipeline.connect("prompt_builder_1.prompt", "string_joiner.strings")
    pipeline.connect("prompt_builder_2.prompt", "string_joiner.strings")

    print(pipeline.run(data={"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}}))

    >> {"string_joiner": {"strings": ["Builder 1: What's Natural Language Processing?", "Builder 2: What is life?"]}}
    ```
    """

    @component.output_types(strings=List[str])
    def run(self, strings: Variadic[str]):
        """
        Joins strings into a list of strings

        :param strings:
            strings from different components

        :returns:
            A dictionary with the following keys:
            - `strings`: Merged list of strings
        """

        out_strings = list(strings)
        return {"strings": out_strings}
