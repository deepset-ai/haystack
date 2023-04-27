from typing import Optional

from haystack.agents.types import Color

STREAMING_CAPABLE_MODELS = ["text-davinci-003", "gpt-3.5-turbo", "gpt-35-turbo", "gpt-4"]


def print_text(text: str, end="", color: Optional[Color] = None) -> None:
    """
    Print text with optional color.
    :param text: Text to print.
    :param end: End character to use (defaults to "").
    :param color: Color to print text in (defaults to None).
    """
    if color:
        print(f"{color.value}{text}{Color.RESET.value}", end=end, flush=True)
    else:
        print(text, end=end, flush=True)
