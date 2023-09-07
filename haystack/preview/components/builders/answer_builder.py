import logging
import re
from typing import List, Dict, Any, Optional

from haystack.preview import component, GeneratedAnswer, Document, default_to_dict, default_from_dict


logger = logging.getLogger(__name__)


@component
class AnswerBuilder:
    """
    A component to parse the output of a Generator to `Answer` objects using regular expressions.
    """

    def __init__(self, pattern: Optional[str] = None, reference_pattern: Optional[str] = None):
        """
        :param pattern: The regular expression pattern to use to extract the answer text from the generator output.
                        If not specified, the whole string is used as the answer. The regular expression can have at
                        most one capture group. If a capture group is present, the text matched by the capture group
                        is used as the answer. If no capture group is present, the whole match is used as the answer.
                        Examples:
                            `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\nthis is an answer".
                            `Answer: (.*)` finds "this is an answer" in a string "this is an argument. Answer: this is an answer".
                        Default: `None`.
        :param reference_pattern: The regular expression pattern to use for parsing the document references.
                                  We assume that references are specified as indices of the input documents and that
                                  indices start at 1.
                                  Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".
                                  If not specified, no parsing is done, and all documents are referenced.
                                  Default: `None`.
        """
        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)

        self.pattern = pattern
        self.reference_pattern = reference_pattern

    @component.output_types(answers=List[List[GeneratedAnswer]])
    def run(
        self,
        queries: List[str],
        replies: List[List[str]],
        metadata: List[List[Dict[str, Any]]],
        documents: Optional[List[List[Document]]] = None,
        pattern: Optional[str] = None,
        reference_pattern: Optional[str] = None,
    ):
        """
        Parse the output of a Generator to `Answer` objects using regular expressions.

        :param queries: The queries used in the prompts for the Generator. A list of strings.
        :param replies: The output of the Generator. A list of lists of strings.
        :param metadata: The metadata returned by the Generator. A list of lists of dictionaries.
        :param documents: The documents used as input to the Generator. A list of lists of `Document` objects. If
                          `documents` are specified, they are added to the `Answer` objects.
                          If both `documents` and `reference_pattern` are specified, the documents referenced in the
                          Generator output are extracted from the input documents and added to the `Answer` objects.
                          Default: `None`.
        :param pattern: The regular expression pattern to use to extract the answer text from the generator output.
                        If not specified, the whole string is used as the answer. The regular expression can have at
                        most one capture group. If a capture group is present, the text matched by the capture group
                        is used as the answer. If no capture group is present, the whole match is used as the answer.
                        Examples:
                            `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\nthis is an answer".
                            `Answer: (.*)` finds "this is an answer" in a string "this is an argument. Answer: this is an answer".
                        Default: `None`.
        :param reference_pattern: The regular expression pattern to use for parsing the document references.
                                  We assume that references are specified as indices of the input documents and that
                                  indices start at 1.
                                  Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".
                                  If not specified, no parsing is done, and all documents are referenced.
                                  Default: `None`.
        """
        if len(queries) != len(replies) != len(metadata):
            raise ValueError(
                f"Number of queries ({len(queries)}), replies ({len(replies)}), and metadata "
                f"({len(metadata)}) must match."
            )

        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)

        documents = documents or []
        pattern = pattern or self.pattern
        reference_pattern = reference_pattern or self.reference_pattern

        all_answers = []
        for i, (query, reply_list, meta_list) in enumerate(zip(queries, replies, metadata)):
            doc_list = documents[i] if i < len(documents) else []

            extracted_answer_strings = AnswerBuilder._extract_answer_strings(reply_list, pattern)

            if doc_list and reference_pattern:
                reference_idxs = AnswerBuilder._extract_reference_idxs(reply_list, reference_pattern)
            else:
                reference_idxs = [[doc_idx for doc_idx, _ in enumerate(doc_list)] for _ in reply_list]

            answers_for_cur_query = []
            for answer_string, doc_idxs, meta in zip(extracted_answer_strings, reference_idxs, meta_list):
                referenced_docs = []
                for idx in doc_idxs:
                    if idx < len(doc_list):
                        referenced_docs.append(doc_list[idx])
                    else:
                        logger.warning("Document index '%s' referenced in Generator output is out of range. ", idx + 1)

                answer = GeneratedAnswer(data=answer_string, query=query, documents=referenced_docs, metadata=meta)
                answers_for_cur_query.append(answer)

            all_answers.append(answers_for_cur_query)

        return {"answers": all_answers}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, pattern=self.pattern, reference_pattern=self.reference_pattern)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerBuilder":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _extract_answer_strings(replies: List[str], pattern: Optional[str] = None) -> List[str]:
        """
        Extract the answer strings from the generator output using the specified pattern.
        If no pattern is specified, the whole string is used as the answer.

        :param replies: The output of the Generator. A list of strings.
        :param pattern: The regular expression pattern to use to extract the answer text from the generator output.
        """
        if pattern is None:
            return replies

        extracted_answers = []
        for reply in replies:
            if match := re.search(pattern, reply):
                # No capture group in pattern -> use the whole match as answer
                if not match.lastindex:
                    extracted_answers.append(match.group(0))
                # One capture group in pattern -> use the capture group as answer
                else:
                    extracted_answers.append(match.group(1))
            else:
                extracted_answers.append("")

        return extracted_answers

    @staticmethod
    def _extract_reference_idxs(replies: List[str], reference_pattern: str) -> List[List[int]]:
        reference_idxs = []
        for reply in replies:
            document_idxs = re.findall(reference_pattern, reply)
            reference_idxs.append([int(idx) - 1 for idx in document_idxs])

        return reference_idxs

    @staticmethod
    def _check_num_groups_in_regex(pattern: str):
        num_groups = re.compile(pattern).groups
        if num_groups > 1:
            raise ValueError(
                f"Pattern '{pattern}' contains multiple capture groups. "
                f"Please specify a pattern with at most one capture group."
            )
