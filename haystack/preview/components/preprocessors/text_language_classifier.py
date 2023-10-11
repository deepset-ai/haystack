import logging
from typing import List, Dict, Any, Optional

from haystack.preview import component, Document, default_from_dict, default_to_dict
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install langdetect'") as langdetect_import:
    import langdetect


@component
class TextLanguageClassifier:
    """
    Routes text inputs onto different output connections depending on their language.
    This is useful for routing queries to different models in a pipeline depending on their language.
    The set of supported languages can be specified.
    For routing Documents based on their language use the related DocumentLanguageClassifier component.
    """

    def __init__(self, languages: Optional[List[str]] = None):
        """
        :param languages: A list of languages in ISO code, each corresponding to a different output connection (see [langdetect` documentation](https://github.com/Mimino666/langdetect#languages)). By default, only ["en"] is supported and texts of any other language are routed to "unmatched".
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages
        component.set_output_types(self, unmatched=List[str], **{language: List[str] for language in languages})

    def run(self, strings: List[str]):
        """
        Run the TextLanguageClassifier. This method routes the strings to different edges based on their language.
        If a string does not match any of the languages specified at initialization, it is routed to
        a connection named "unmatched".

        :param strings: A list of strings to route to different edges.
        """
        if not isinstance(strings, list) or strings and not isinstance(strings[0], str):
            raise TypeError(
                "TextLanguageClassifier expects a list of str as input. In case you want to classify a document, please use the DocumentLanguageClassifier."
            )

        output: Dict[str, List[str]] = {language: [] for language in self.languages}
        output["unmatched"] = []

        for string in strings:
            detected_language = self.detect_language(string)
            if detected_language in self.languages:
                output[detected_language].append(string)
            else:
                output["unmatched"].append(string)

        return output

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, languages=self.languages)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextLanguageClassifier":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def detect_language(self, string: str) -> Optional[str]:
        try:
            language = langdetect.detect(string)
        except langdetect.LangDetectException:
            logger.warning("Langdetect cannot detect the language of text: %s", string)
            language = None
        return language
