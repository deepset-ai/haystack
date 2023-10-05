import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from haystack.preview import Document, component, default_from_dict, default_to_dict


@component
class TextDocumentCleaner:
    """
    Makes text documents more readable by cleaning empty lines, extra whitespaces, headers and footers, etc.
    This is useful for preparing the documents for further processing by LLMs.
    """

    def __init__(
        self,
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
        remove_substrings: Optional[List[str]] = None,
        remove_regex: Optional[str] = None,
    ):
        """
        :param remove_empty_lines: Whether to remove empty lines.
        :param remove_extra_whitespaces: Whether to remove extra whitespaces.
        :param remove_repeated_substrings: Whether to remove repeated substrings, such as headers and footers.
        :param remove_substrings: List of substrings to remove from the text.
        :param remove_regex: Regex to match and replace substrings by "".
        """

        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_repeated_substrings = remove_repeated_substrings
        self.remove_substrings = remove_substrings
        self.remove_regex = remove_regex

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("TextDocumentCleaner expects a List of Documents as input.")

        cleaned_docs = []
        for doc in documents:
            if doc.text is None:
                raise ValueError(
                    f"TextDocumentCleaner only works with text documents but document.text for document ID {doc.id} is None."
                )
            text = doc.text

            if self.remove_empty_lines:
                text = self._remove_empty_lines(text)
            if self.remove_extra_whitespaces:
                text = self._remove_extra_whitespaces(text)
            if self.remove_repeated_substrings:
                text = self._remove_repeated_substrings(text)
            if self.remove_substrings:
                text = self._remove_substrings(text, self.remove_substrings)
            if self.remove_regex:
                text = self._remove_regex(text, self.remove_regex)

            cleaned_docs.append(Document(text=text, metadata=deepcopy(doc.metadata)))

        return {"documents": cleaned_docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            clean_empty_lines=self.remove_empty_lines,
            clean_whitespaces=self.remove_extra_whitespaces,
            clean_repeated_substrings=self.remove_repeated_substrings,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextDocumentCleaner":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def _remove_empty_lines(self, text: str) -> str:
        """
        Remove empty lines and lines that contain nothing but whitespaces from text.
        :param text: Text to clean.
        """
        lines = text.split("\n")
        non_empty_lines = filter(lambda line: line.strip() != "", lines)
        return "\n".join(non_empty_lines)

    def _remove_extra_whitespaces(self, text: str) -> str:
        """
        Remove extra whitespaces from text.
        :param text: Text to clean.
        """
        return re.sub(r"\s\s+", " ", text).strip()

    def _remove_regex(self, text: str, regex: str) -> str:
        """
        Remove substrings that match the specified regex from the text.
        :param text: Text to clean.
        :param regex: Regex to match and replace substrings by "".
        """
        return re.sub(regex, "", text).strip()

    def _remove_substrings(self, text: str, substrings: List[str]) -> str:
        """
        Remove all specified substrings from the text.
        :param text: Text to clean.
        :param substrings: Substrings to remove.
        """
        for substring in substrings:
            text = text.replace(substring, "")
        return text

    def _remove_repeated_substrings(self, text: str) -> str:
        return text
