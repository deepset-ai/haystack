from typing import List, Dict, Any, Optional

from haystack.preview import component, Document, default_from_dict, default_to_dict


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
        :param languages: The languages that are supported as output edges. By default english is supported and texts of any other language are routed to "unmatched"
        """
        if not languages:
            languages = ["english"]
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

        unmatched_strings = []
        output: Dict[str, List[str]] = {language: [] for language in self.languages}

        for string in strings:
            cur_string_matched = False
            for language in self.languages:
                if self.string_matches_language(language, string):
                    output[language].append(string)
                    cur_string_matched = True
                    break

            if not cur_string_matched:
                unmatched_strings.append(string)

        output["unmatched"] = unmatched_strings
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

    def string_matches_language(self, language: str, string: str) -> bool:
        return language == "english"
