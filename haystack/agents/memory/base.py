from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Union

from haystack.nodes import PromptNode, PromptTemplate


class Memory(ABC):
    """
    Abstract base class for memory management in an Agent.
    """

    @abstractmethod
    def load(self, keys: Optional[List[str]] = None, **kwargs) -> Any:
        """
        Load the context of this model run from memory.

        :param keys: Optional list of keys to specify the data to load.
        :return: The loaded data.
        """

    @abstractmethod
    def save(self, data: Dict[str, Any]) -> None:
        """
        Save the context of this model run to memory.

        :param data: A dictionary containing the data to save.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Clear memory contents.
        """


class NoMemory(Memory):
    """
    A memory class that doesn't store any data.
    """

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> Any:
        """
        Load an empty dictionary.

        :param keys: Optional list of keys (ignored in this implementation).
        :return: An empty dictionary.
        """
        return {}

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save method that does nothing.

        :param data: A dictionary containing the data to save (ignored in this implementation).
        """
        pass

    def clear(self) -> None:
        """
        Clear method that does nothing.
        """
        pass


class ConversationMemory(Memory):
    """
    A memory class that stores conversation history.
    """

    def __init__(self, input_key: str = "input", output_key: str = "output"):
        """
        Initialize ConversationMemory with input and output keys.

        :param input_key: The key to use for storing user input.
        :param output_key: The key to use for storing model output.
        """
        self.list: List[OrderedDict] = []
        self.input_key = input_key
        self.output_key = output_key

    def load(self, keys: Optional[List[str]] = None, k: Optional[int] = None, **kwargs) -> Any:
        """
        Load conversation history as a formatted string.

        :param keys: Optional list of keys (ignored in this implementation).
        :param k: Optional integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history.
        """
        chat_transcript = ""

        if k is not None:
            chat_list = self.list[-k:]  # pylint: disable=invalid-unary-operand-type
        else:
            chat_list = self.list

        for chat_snippet in chat_list:
            chat_transcript += f"Human: {chat_snippet['Human']}\n"
            chat_transcript += f"AI: {chat_snippet['AI']}\n"
        return chat_transcript

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a conversation snippet to memory.

        :param data: A dictionary containing the conversation snippet to save.
        """
        chat_snippet = OrderedDict()
        chat_snippet["Human"] = data[self.input_key]
        chat_snippet["AI"] = data[self.output_key]
        self.list.append(chat_snippet)

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.list = []


class ConversationSummaryMemory(ConversationMemory):
    """
    A memory class that stores conversation history and periodically generates summaries.
    """

    def __init__(
        self,
        pn: PromptNode,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        input_key: str = "input",
        output_key: str = "output",
        summary_frequency: int = 3,
    ):
        """
        Initialize ConversationSummaryMemory with a PromptNode, optional prompt_template,
        input and output keys, and a summary_frequency.

        :param pn: A PromptNode object for generating conversation summaries.
        :param prompt_template: Optional prompt template as a string or PromptTemplate object.
        :param input_key: input key, default is "input".
        :param output_key: output key, default is "output".
        :param summary_frequency: integer specifying how often to generate a summary (default is 3).
        """
        super().__init__(input_key, output_key)
        self.save_count = 0
        self.pn = pn

        template = (
            pn.default_prompt_template
            if pn.default_prompt_template is not None
            else prompt_template or "conversational-summary"
        )
        self.template = pn.get_prompt_template(template)
        self.summary_frequency = summary_frequency
        self.summary = ""

    def load(self, keys: Optional[List[str]] = None, k: Optional[int] = None, **kwargs) -> Any:
        """
        Load conversation history as a formatted string, including the latest summary.

        :param keys: Optional list of keys (ignored in this implementation).
        :param k: Optional integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history with the latest summary.
        """
        return f"{self.summary}\n {super().load(keys, k, **kwargs)}"

    def summarize(self) -> str:
        """
        Generate a summary of the conversation history and clear the history.

        :return: A string containing the generated summary.
        """
        chat_transcript = self.load()
        self.clear()
        return self.pn.prompt(self.template, chat_transcript=chat_transcript)[0]

    def needs_summary(self) -> bool:
        """
        Determine if a new summary should be generated.

        :return: True if a new summary should be generated, otherwise False.
        """
        return self.save_count % self.summary_frequency == 0

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a conversation snippet to memory and update the save count.
        Generate a summary if needed.

        :param data: A dictionary containing the conversation snippet to save.
        """
        super().save(data)
        self.save_count += 1
        if self.needs_summary():
            self.summary = self.summarize()

    def clear(self) -> None:
        """
        Clear the conversation history and the summary.
        """
        super().clear()
        self.summary = ""
