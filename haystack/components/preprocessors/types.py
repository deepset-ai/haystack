import re
from typing import Literal

Language = Literal[
    "ru", "sl", "es", "sv", "tr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "it", "no", "pl", "pt", "ml"
]
ISO639_TO_NLTK = {
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ml": "malayalam",
}

QUOTE_SPANS_RE = re.compile(r"\W(\"+|\'+).*?\1")
