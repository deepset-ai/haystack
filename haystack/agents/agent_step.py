from __future__ import annotations

import logging
import re
from typing import Optional, Dict, Tuple, Any

from haystack import Answer
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
        final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
        prompt_node_response: str = "",
        transcript: str = "",
    ):
        """
        :param current_step: The current step in the execution of the agent.
        :param max_steps: The maximum number of steps the agent can execute.
        :param final_answer_pattern: The regex pattern to extract the final answer from the PromptNode response.
        :param prompt_node_response: The PromptNode response received.
        :param transcript: The full Agent execution transcript based on the Agent's initial prompt template and the
        text it generated during execution up to this step. The transcript is used to generate the next prompt.
        """
        self.current_step = current_step
        self.max_steps = max_steps
        self.final_answer_pattern = final_answer_pattern
        self.prompt_node_response = prompt_node_response
        self.transcript = transcript

    def prepare_prompt(self):
        """
        Prepares the prompt for the next step.
        """
        return self.transcript

    def create_next_step(self, prompt_node_response: Any) -> AgentStep:
        """
        Creates the next agent step based on the current step and the PromptNode response.
        :param prompt_node_response: The PromptNode response received.
        """
        if not isinstance(prompt_node_response, list):
            raise AgentError(
                f"Agent output must be a list of str, but {prompt_node_response} received. "
                f"Transcript:\n{self.transcript}"
            )

        if not prompt_node_response:
            raise AgentError(
                f"Agent output must be a non empty list of str, but {prompt_node_response} received. "
                f"Transcript:\n{self.transcript}"
            )

        return AgentStep(
            current_step=self.current_step + 1,
            max_steps=self.max_steps,
            final_answer_pattern=self.final_answer_pattern,
            prompt_node_response=prompt_node_response[0],
            transcript=self.transcript,
        )

    def extract_tool_name_and_tool_input(self, tool_pattern: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the tool name and the tool input from the PromptNode response.
        :param tool_pattern: The regex pattern to extract the tool name and the tool input from the PromptNode response.
        :return: A tuple containing the tool name and the tool input.
        """
        tool_match = re.search(tool_pattern, self.prompt_node_response)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_input = tool_match.group(3)
            return tool_name.strip('" []\n').strip(), tool_input.strip('" \n')
        return None, None

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
        }
        if self.current_step >= self.max_steps:
            logger.warning(
                "Maximum number of iterations (%s) reached for query (%s). Increase max_steps "
                "or no answer can be provided for this query.",
                self.max_steps,
                query,
            )
        else:
            final_answer = self.extract_final_answer()
            if not final_answer:
                logger.warning(
                    "Final answer pattern (%s) not found in PromptNode response (%s).",
                    self.final_answer_pattern,
                    self.prompt_node_response,
                )
            else:
                answer = {
                    "query": query,
                    "answers": [Answer(answer=final_answer, type="generative")],
                    "transcript": self.transcript,
                }
        return answer

    def extract_final_answer(self) -> Optional[str]:
        """
        Parse the final answer from the PromptNode response.
        :return: The final answer.
        """
        if not self.is_last():
            raise AgentError("Cannot extract final answer from non terminal step.")

        final_answer_match = re.search(self.final_answer_pattern, self.prompt_node_response)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            return final_answer.strip('" ')
        return None

    def is_final_answer_pattern_found(self) -> bool:
        """
        Check if the final answer pattern was found in PromptNode response.
        :return: True if the final answer pattern was found in PromptNode response, False otherwise.
        """
        return bool(re.search(self.final_answer_pattern, self.prompt_node_response))

    def is_last(self) -> bool:
        """
        Check if this is the last step of the Agent.
        :return: True if this is the last step of the Agent, False otherwise.
        """
        return self.is_final_answer_pattern_found() or self.current_step >= self.max_steps

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
