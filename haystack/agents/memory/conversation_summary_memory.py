from typing import Optional, Union, Dict, Any, List

from haystack.agents.memory import ConversationMemory
from haystack.nodes import PromptTemplate, PromptNode


class ConversationSummaryMemory(ConversationMemory):
    """
    A memory class that stores conversation history and periodically generates summaries.
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        input_key: str = "input",
        output_key: str = "output",
        summary_frequency: int = 3,
    ):
        """
        Initialize ConversationSummaryMemory with a PromptNode, optional prompt_template,
        input and output keys, and a summary_frequency.

        :param prompt_node: A PromptNode object for generating conversation summaries.
        :param prompt_template: Optional prompt template as a string or PromptTemplate object.
        :param input_key: input key, default is "input".
        :param output_key: output key, default is "output".
        :param summary_frequency: integer specifying how often to generate a summary (default is 3).
        """
        super().__init__(input_key, output_key)
        self.save_count = 0
        self.prompt_node = prompt_node

        template = (
            prompt_template
            if prompt_template is not None
            else prompt_node.default_prompt_template or "conversational-summary"
        )
        self.template = prompt_node.get_prompt_template(template)
        self.summary_frequency = summary_frequency
        self.summary = ""

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
        """
        Load conversation history as a formatted string, including the latest summary.

        :param keys: Optional list of keys (ignored in this implementation).
        :param kwargs: Optional keyword arguments
            - window_size: integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history with the latest summary.
        """
        if self.has_unsummarized_snippets():
            unsummarized = self.load_recent_snippets(window_size=self.unsummarized_snippets())
            return f"{self.summary}\n{unsummarized}"
        else:
            return self.summary

    def load_recent_snippets(self, window_size: int = 1) -> str:
        """
        Load the most recent conversation snippets as a formatted string.

        :param window_size: integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the most recent conversation snippets.
        """
        return super().load(window_size=window_size)

    def summarize(self) -> str:
        """
        Generate a summary of the conversation history and clear the history.

        :return: A string containing the generated summary.
        """
        most_recent_chat_snippets = self.load_recent_snippets(window_size=self.summary_frequency)
        pn_response = self.prompt_node.prompt(self.template, chat_transcript=most_recent_chat_snippets)
        return pn_response[0]

    def needs_summary(self) -> bool:
        """
        Determine if a new summary should be generated.

        :return: True if a new summary should be generated, otherwise False.
        """
        return self.save_count % self.summary_frequency == 0

    def unsummarized_snippets(self) -> int:
        """
        Returns how many conversation snippets have not been summarized.
        :return: The number of conversation snippets that have not been summarized.
        """
        return self.save_count % self.summary_frequency

    def has_unsummarized_snippets(self) -> bool:
        """
        Returns True if there are any conversation snippets that have not been summarized.
        :return: True if there are unsummarized snippets, otherwise False.
        """
        return self.unsummarized_snippets() != 0

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a conversation snippet to memory and update the save count.
        Generate a summary if needed.

        :param data: A dictionary containing the conversation snippet to save.
        """
        super().save(data)
        self.save_count += 1
        if self.needs_summary():
            self.summary += self.summarize()

    def clear(self) -> None:
        """
        Clear the conversation history and the summary.
        """
        super().clear()
        self.save_count = 0
        self.summary = ""
