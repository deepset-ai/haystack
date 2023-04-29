from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from haystack import Answer
from haystack.agents.answer_parsers import AgentAnswerParser
from haystack.errors import AgentError

logger = logging.getLogger(__name__)


class AgentStep:
    """
    The AgentStep class represents a single step in the execution of an agent.

    """

    def __init__(
        self,
        current_step: int = 1,
        max_steps: int = 10,
        final_answer_parser: AgentAnswerParser = None,
        prompt_node_response: str = "",
        transcript: str = "",
    ):
        """
        :param current_step: The current step in the execution of the agent.
        :param max_steps: The maximum number of steps the agent can execute.
        :param final_answer_parser: AgentAnswerParser to extract the final answer from the PromptNode response.
        :param prompt_node_response: The PromptNode response received.
        text it generated during execution up to this step. The transcript is used to generate the next prompt.
        """
        self.current_step = current_step
        self.max_steps = max_steps
        self.final_answer_parser = final_answer_parser
        self.prompt_node_response = prompt_node_response
        self.transcript = transcript
        self.tool_results = []

    def create_next_step(self, prompt_node_response: Any, current_step: Optional[int] = None) -> AgentStep:
        """
        Creates the next agent step based on the current step and the PromptNode response.
        :param prompt_node_response: The PromptNode response received.
        :param current_step: The current step in the execution of the agent.
        """
        if not isinstance(prompt_node_response, list) or not prompt_node_response:
            raise AgentError(
                f"Agent output must be a non-empty list of str, but {prompt_node_response} received. "
                f"Transcript:\n{self.transcript}"
            )
        cls = type(self)
        return cls(
            current_step=current_step if current_step else self.current_step + 1,
            max_steps=self.max_steps,
            final_answer_parser=self.final_answer_parser,
            prompt_node_response=prompt_node_response[0],
            transcript=self.transcript,
        )

    def final_answer(self, query: str) -> Dict[str, Any]:
        """
        Formats an answer as a dict containing `query` and `answers` similar to the output of a Pipeline.
        The full transcript based on the Agent's initial prompt template and the text it generated during execution.

        :param query: The search query
        """
        answer: Dict[str, Any] = {
            "query": query,
            "answers": [Answer(answer="", type="generative")],
            "transcript": self.transcript,
            "tool_results": self.tool_results
        }
        if self.current_step > self.max_steps:
            logger.warning(
                "Maximum number of iterations (%s) reached for query (%s). Increase max_steps "
                "or no answer can be provided for this query.",
                self.max_steps,
                query,
            )
        else:
            final_answer = self.final_answer_parser.parse(self.prompt_node_response)
            if not final_answer:
                logger.warning(
                    "Final answer parser (%s) could not parse PromptNode response (%s).",
                    self.final_answer_parser,
                    self.prompt_node_response,
                )
            else:
                answer = {
                    "query": query,
                    "answers": [Answer(answer=final_answer, type="generative")],
                    "transcript": self.transcript,
                    "tool_results": self.tool_results
                }
        return answer

    def is_last(self) -> bool:
        """
        Check if this is the last step of the Agent.
        :return: True if this is the last step of the Agent, False otherwise.
        """
        return self.final_answer_parser.can_parse(self.prompt_node_response) or self.current_step > self.max_steps

    def completed(self, observation: Optional[str]):
        """
        Update the transcript with the observation
        :param observation: received observation from the Agent environment.
        """
        self.transcript += (
            f"{self.prompt_node_response}\nObservation: {observation}\nThought:"
            if observation
            else self.prompt_node_response
        )

    def __repr__(self):
        return (
            f"AgentStep(current_step={self.current_step}, max_steps={self.max_steps}, "
            f"prompt_node_response={self.prompt_node_response}, transcript={self.transcript})"
        )
