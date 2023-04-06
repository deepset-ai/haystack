from typing import Optional, List, Union

from haystack.schema import Answer, Document

from haystack.nodes.other.shaper import (  # pylint: disable=unused-import
    Shaper,
    join_documents_to_string as join,  # used as shaping function
    format_document,
    format_answer,
    format_string,
)


def to_strings(items: List[Union[str, Document, Answer]], pattern=None, str_replace=None) -> List[str]:
    results = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            results.append(format_string(item, str_replace=str_replace))
        elif isinstance(item, Document):
            results.append(format_document(document=item, pattern=pattern, str_replace=str_replace, idx=idx))
        elif isinstance(item, Answer):
            results.append(format_answer(answer=item, pattern=pattern, str_replace=str_replace, idx=idx))
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")
    return results


class BaseOutputParser(Shaper):
    """
    An output parser in `PromptTemplate` defines how to parse the model output and convert it into Haystack primitives (answers, documents, or labels).
    BaseOutputParser is the base class for output parser implementations.
    """

    @property
    def output_variable(self) -> Optional[str]:
        return self.outputs[0]


class AnswerParser(BaseOutputParser):
    """
    Parses the model output to extract the answer into a proper `Answer` object using regex patterns.
    AnswerParser adds the `document_ids` of the documents used to generate the answer and the prompts used to the `Answer` object.
    You can pass a `reference_pattern` to extract the document_ids of the answer from the model output.
    """

    def __init__(self, pattern: Optional[str] = None, reference_pattern: Optional[str] = None):
        """
         :param pattern: The regex pattern to use for parsing the answer.
            Examples:
                `[^\\n]+$` finds "this is an answer" in string "this is an argument.\nthis is an answer".
                `Answer: (.*)` finds "this is an answer" in string "this is an argument. Answer: this is an answer".
            If not specified, the whole string is used as the answer. If specified, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.
        :param reference_pattern: The regex pattern to use for parsing the document references.
            Example: `\\[(\\d+)\\]` finds "1" in string "this is an answer[1]".
            If None, no parsing is done and all documents are referenced.
        """
        self.pattern = pattern
        self.reference_pattern = reference_pattern
        super().__init__(
            func="strings_to_answers",
            inputs={"strings": "results"},
            outputs=["answers"],
            params={"pattern": pattern, "reference_pattern": reference_pattern},
        )
