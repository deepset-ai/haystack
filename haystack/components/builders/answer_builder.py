# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import replace
from typing import Any, Optional, Union

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

    ### Usage example with documents and reference pattern

    ```python
    from haystack import Document
    from haystack.components.builders import AnswerBuilder

    replies = ["The capital of France is Paris [2]."]

    docs = [
        Document(content="Berlin is the capital of Germany."),
        Document(content="Paris is the capital of France."),
        Document(content="Rome is the capital of Italy."),
    ]

    builder = AnswerBuilder(reference_pattern="\\[(\\d+)\\]", return_only_referenced_documents=False)
    result = builder.run(query="What is the capital of France?", replies=replies, documents=docs)["answers"][0]

    print(f"Answer: {result.data}")
    print("References:")
    for doc in result.documents:
        if doc.meta["referenced"]:
            print(f"[{doc.meta['source_index']}] {doc.content}")
    print("Other sources:")
    for doc in result.documents:
        if not doc.meta["referenced"]:
            print(f"[{doc.meta['source_index']}] {doc.content}")

    # Answer: The capital of France is Paris
    # References:
    # [2] Paris is the capital of France.
    # Other sources:
    # [1] Berlin is the capital of Germany.
    # [3] Rome is the capital of Italy.
    ```
    """

    def __init__(
        self,
        pattern: Optional[str] = None,
        reference_pattern: Optional[str] = None,
        last_message_only: bool = False,
        *,
        return_only_referenced_documents: bool = True,
    ):
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
            If not specified, no parsing is done, and all documents are returned.
            References need to be specified as indices of the input documents and start at [1].
            Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".
            If this parameter is provided, documents metadata will contain a "referenced" key with a boolean value.

        :param last_message_only:
           If False (default value), all messages are used as the answer.
           If True, only the last message is used as the answer.

        :param return_only_referenced_documents:
            To be used in conjunction with `reference_pattern`.
            If True (default value), only the documents that were actually referenced in `replies` are returned.
            If False, all documents are returned.
            If `reference_pattern` is not provided, this parameter has no effect, and all documents are returned.
        """
        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)

        self.pattern = pattern
        self.reference_pattern = reference_pattern
        self.last_message_only = last_message_only
        self.return_only_referenced_documents = return_only_referenced_documents

    @component.output_types(answers=list[GeneratedAnswer])
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        query: str,
        replies: Union[list[str], list[ChatMessage]],
        meta: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[Document]] = None,
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
            the `GeneratedAnswer` objects.
            Each Document.meta includes a "source_index" key, representing its 1-based position in the input list.
            When `reference_pattern` is provided:
            - "referenced" key is added to the Document.meta, indicating if the document was referenced in the output.
            - `return_only_referenced_documents` init parameter controls if all or only referenced documents are
            returned.
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
            If not specified, no parsing is done, and all documents are returned.
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

        replies_to_iterate = replies[-1:] if self.last_message_only and replies else replies
        meta_to_iterate = meta[-1:] if self.last_message_only and meta else meta

        all_answers = []
        for reply, given_metadata in zip(replies_to_iterate, meta_to_iterate):
            # Extract content from ChatMessage objects if reply is a ChatMessages, else use the string as is
            extracted_reply = reply.text or "" if isinstance(reply, ChatMessage) else str(reply)
            extracted_metadata = reply.meta if isinstance(reply, ChatMessage) else {}

            extracted_metadata = {**extracted_metadata, **given_metadata}
            extracted_metadata["all_messages"] = replies

            referenced_docs = []
            if documents:
                referenced_idxs = (
                    AnswerBuilder._extract_reference_idxs(extracted_reply, reference_pattern)
                    if reference_pattern
                    else set()
                )
                doc_idxs = (
                    referenced_idxs
                    if reference_pattern and self.return_only_referenced_documents
                    else set(range(len(documents)))
                )

                for idx in doc_idxs:
                    try:
                        doc = documents[idx]
                    except IndexError:
                        logger.warning(
                            "Document index '{index}' referenced in Generator output is out of range. ", index=idx + 1
                        )
                        continue

                    doc_meta: dict[str, Any] = doc.meta or {}
                    doc_meta["source_index"] = idx + 1
                    if reference_pattern:
                        doc_meta["referenced"] = idx in referenced_idxs
                    referenced_docs.append(replace(doc, meta=doc_meta))

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
    def _extract_reference_idxs(reply: str, reference_pattern: str) -> set[int]:
        document_idxs = re.findall(reference_pattern, reply)
        return {int(idx) - 1 for idx in document_idxs}

    @staticmethod
    def _check_num_groups_in_regex(pattern: str):
        num_groups = re.compile(pattern).groups
        if num_groups > 1:
            raise ValueError(
                f"Pattern '{pattern}' contains multiple capture groups. "
                f"Please specify a pattern with at most one capture group."
            )
