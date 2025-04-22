# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Dict, List, Optional, Union

from haystack import Document, GeneratedAnswer, component, logging
from haystack.dataclasses.chat_message import ChatMessage

logger = logging.getLogger(__name__)


@component
class AnswerBuilder:
    """
    Converts a query and Generator replies into a `GeneratedAnswer` object.

    AnswerBuilder parses Generator replies using custom regular expressions.
    Check out the usage example below to see how it works.
    Optionally, it can also take documents and metadata from the Generator to add to the `GeneratedAnswer` object.
    AnswerBuilder works with both non-chat and chat Generators.

    ### Usage example

    ```python
    from haystack.components.builders import AnswerBuilder

    builder = AnswerBuilder(pattern="Answer: (.*)")
    builder.run(query="What's the answer?", replies=["This is an argument. Answer: This is the answer."])
    ```
    """

    def __init__(self, pattern: Optional[str] = None, reference_pattern: Optional[str] = None):
        """
        Creates an instance of the AnswerBuilder component.

        :param pattern:
            The regular expression pattern to extract the answer text from the Generator.
            If not specified, the entire response is used as the answer.
            The regular expression can have one capture group at most.
            If present, the capture group text
            is used as the answer. If no capture group is present, the whole match is used as the answer.
            Examples:
                `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\\nthis is an answer".
                `Answer: (.*)` finds "this is an answer" in a string "this is an argument. Answer: this is an answer".

        :param reference_pattern:
            The regular expression pattern used for parsing the document references.
            If not specified, no parsing is done, and all documents are referenced.
            References need to be specified as indices of the input documents and start at [1].
            Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".
        """
        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)

        self.pattern = pattern
        self.reference_pattern = reference_pattern

    @component.output_types(answers=List[GeneratedAnswer])
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        query: str,
        replies: Union[List[str], List[ChatMessage]],
        meta: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Document]] = None,
        pattern: Optional[str] = None,
        reference_pattern: Optional[str] = None,
    ):
        """
        Turns the output of a Generator into `GeneratedAnswer` objects using regular expressions.

        :param query:
            The input query used as the Generator prompt.
        :param replies:
            The output of the Generator. Can be a list of strings or a list of `ChatMessage` objects.
        :param meta:
            The metadata returned by the Generator. If not specified, the generated answer will contain no metadata.
        :param documents:
            The documents used as the Generator inputs. If specified, they are added to
            the`GeneratedAnswer` objects.
            If both `documents` and `reference_pattern` are specified, the documents referenced in the
            Generator output are extracted from the input documents and added to the `GeneratedAnswer` objects.
        :param pattern:
            The regular expression pattern to extract the answer text from the Generator.
            If not specified, the entire response is used as the answer.
            The regular expression can have one capture group at most.
            If present, the capture group text
            is used as the answer. If no capture group is present, the whole match is used as the answer.
                Examples:
                    `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\\nthis is an answer".
                    `Answer: (.*)` finds "this is an answer" in a string
                    "this is an argument. Answer: this is an answer".
        :param reference_pattern:
            The regular expression pattern used for parsing the document references.
            If not specified, no parsing is done, and all documents are referenced.
            References need to be specified as indices of the input documents and start at [1].
            Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".

        :returns: A dictionary with the following keys:
            - `answers`: The answers received from the output of the Generator.
        """
        if not meta:
            meta = [{}] * len(replies)
        elif len(replies) != len(meta):
            raise ValueError(f"Number of replies ({len(replies)}), and metadata ({len(meta)}) must match.")

        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)

        pattern = pattern or self.pattern
        reference_pattern = reference_pattern or self.reference_pattern
        all_answers = []
        for reply, given_metadata in zip(replies, meta):
            # Extract content from ChatMessage objects if reply is a ChatMessages, else use the string as is
            if isinstance(reply, ChatMessage):
                if reply.text is None:
                    raise ValueError(f"The provided ChatMessage has no text. ChatMessage: {reply}")
                extracted_reply = reply.text
            else:
                extracted_reply = str(reply)
            extracted_metadata = reply.meta if isinstance(reply, ChatMessage) else {}

            extracted_metadata = {**extracted_metadata, **given_metadata}

            referenced_docs = []
            if documents:
                if reference_pattern:
                    reference_idxs = AnswerBuilder._extract_reference_idxs(extracted_reply, reference_pattern)
                else:
                    reference_idxs = [doc_idx for doc_idx, _ in enumerate(documents)]

                for idx in reference_idxs:
                    try:
                        referenced_docs.append(documents[idx])
                    except IndexError:
                        logger.warning(
                            "Document index '{index}' referenced in Generator output is out of range. ", index=idx + 1
                        )

            answer_string = AnswerBuilder._extract_answer_string(extracted_reply, pattern)
            answer = GeneratedAnswer(
                data=answer_string, query=query, documents=referenced_docs, meta=extracted_metadata
            )
            all_answers.append(answer)

        return {"answers": all_answers}

    @staticmethod
    def _extract_answer_string(reply: str, pattern: Optional[str] = None) -> str:
        """
        Extract the answer string from the generator output using the specified pattern.

        If no pattern is specified, the whole string is used as the answer.

        :param reply:
            The output of the Generator. A string.
        :param pattern:
            The regular expression pattern to use to extract the answer text from the generator output.
        """
        if pattern is None:
            return reply

        if match := re.search(pattern, reply):
            # No capture group in pattern -> use the whole match as answer
            if not match.lastindex:
                return match.group(0)
            # One capture group in pattern -> use the capture group as answer
            return match.group(1)
        return ""

    @staticmethod
    def _extract_reference_idxs(reply: str, reference_pattern: str) -> List[int]:
        document_idxs = re.findall(reference_pattern, reply)
        return [int(idx) - 1 for idx in document_idxs]

    @staticmethod
    def _check_num_groups_in_regex(pattern: str):
        num_groups = re.compile(pattern).groups
        if num_groups > 1:
            raise ValueError(
                f"Pattern '{pattern}' contains multiple capture groups. "
                f"Please specify a pattern with at most one capture group."
            )
