# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# the following lines avoid errors in the CI
# ruff: noqa: E501
# mypy: ignore-errors

from haystack.components.converters import DOCXToDocument
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.extractors import NamedEntityExtractor
from haystack.components.generators.chat.hugging_face_api import HuggingFaceAPIChatGenerator
from haystack.components.generators.chat.hugging_face_local import HuggingFaceLocalChatGenerator
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator


class ChatGeneratorEncapsulator:
    def __init__(self, chat_generator: ChatGenerator):
        self.chat_generator = chat_generator

    def show(self):
        """
        Print the dictionary representation of the chat generator.
        """
        print(self.chat_generator.to_dict())


my_chat_generator = ChatGeneratorEncapsulator(OpenAIChatGenerator())  # mypy: OK
my_chat_generator.show()  # works

another_chat_generator = ChatGeneratorEncapsulator(DOCXToDocument())  # mypy: Error
# error: Argument 1 to "ChatGeneratorEncapsulator" has incompatible type "DOCXToDocument"; expected "ChatGenerator"  [arg-type]
# note: Following member(s) of "DOCXToDocument" have conflicts:
# note:     Expected:
# note:         def run(self, messages: list[ChatMessage]) -> dict[str, Any]
# note:     Got:
# note:         def run(sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = ...) -> Any

another_chat_generator.show()  # works


another_cg2 = ChatGeneratorEncapsulator(OpenAITextEmbedder())  # mypy: Error
another_cg2.show()  # works


another_cg3 = ChatGeneratorEncapsulator(HuggingFaceLocalChatGenerator())  # mypy: OK
another_cg3.show()  # works


another_cg4 = ChatGeneratorEncapsulator(NamedEntityExtractor(backend="spacy", model="en_core_web_sm"))  # mypy: error
another_cg4.show()  # works


another_cg5 = ChatGeneratorEncapsulator(
    HuggingFaceAPIChatGenerator(api_type="something", api_params={"model": "some-model"})
)  # mypy: OK
another_cg5.show()  # works
