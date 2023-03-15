from __future__ import annotations

import logging
import re
from typing import List, Optional, Union, Dict, Tuple, Any

from haystack import Pipeline, BaseComponent, Answer, Document
from haystack.errors import AgentError
from haystack.nodes import PromptNode, BaseRetriever, PromptTemplate
from haystack.pipelines import (
    BaseStandardPipeline,
    ExtractiveQAPipeline,
    DocumentSearchPipeline,
    GenerativeQAPipeline,
    SearchSummarizationPipeline,
    FAQPipeline,
    TranslationWrapperPipeline,
    RetrieverQuestionGenerationPipeline,
)
from haystack.telemetry import send_custom_event

logger = logging.getLogger(__name__)


class AgentStep:
    def __init__(
        self,
        current_step: int = 1,
        max_steps: int = 8,
        final_answer_pattern: str = "",
        llm_response: str = "",
        transcript: str = "",
    ):
        self.current_step = current_step
        self.max_steps = max_steps
        self.final_answer_pattern = final_answer_pattern
        self.llm_response = llm_response
        self.transcript = transcript

    def prepare_prompt(self):
        return self.transcript

    def create_next_step(self, prompt_node_response: Any) -> AgentStep:
        return AgentStep(
            current_step=self.current_step + 1,
            max_steps=self.max_steps,
            final_answer_pattern=self.final_answer_pattern,
            llm_response=prompt_node_response[0],
            transcript=self.transcript,
        )

    def extract_tool_name_and_tool_input(self, tool_pattern: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the tool name and the tool input from the prediction output of the Agent's PromptNode.
        """
        tool_match = re.search(tool_pattern, self.llm_response)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_input = tool_match.group(3)
            return tool_name.strip('" []').strip(), tool_input.strip('" ')
        return None, None

    def final_answer(self, query: str) -> Dict[str, Union[str, List[Answer]]]:
        """
        Formats an answer as a dict containing `query` and `answers` similar to the output of a Pipeline.
        The full transcript based on the Agent's initial prompt template and the text it generated during execution.

        :param query: The search query.
        :param answer: The final answer returned by the Agent. An empty string corresponds to no answer.
        :param transcript: The text generated by the Agent and the initial filled template for debug purposes.
        """
        answer: Dict[str, Union[str, List[Answer]]] = {}
        if self.current_step >= self.max_steps:
            logger.warning(
                "Maximum number of iterations (%s) reached for query (%s). Increase max_iterations "
                "or no answer can be provided for this query.",
                self.max_steps,
                query,
            )
        else:
            final_answer = self.extract_final_answer()
            answer = {
                "query": query,
                "answers": [Answer(answer=final_answer, type="generative")],
                "transcript": self.transcript,
            }
        return answer

    def extract_final_answer(self) -> str:
        """
        Parse the final answer from the prediction output of the Agent's PromptNode.
        """
        final_answer = ""
        final_answer_match = re.search(self.final_answer_pattern, self.llm_response)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            final_answer = final_answer.strip('" ')
        return final_answer

    def is_final_answer_pattern_found(self) -> bool:
        """
        Check if the final answer pattern was found in the prediction output of the Agent's PromptNode.
        """
        return bool(re.search(self.final_answer_pattern, self.llm_response))

    def is_terminal(self) -> bool:
        """
        Check if the Agent's PromptNode has reached a terminal state.

        """
        return self.is_final_answer_pattern_found() or self.current_step >= self.max_steps

    def completed(self, observation: Optional[str] = None):
        if observation:
            self.transcript += f"{self.llm_response}\nObservation: {observation}\nThought:"
        else:
            self.transcript = self.llm_response


class Tool:
    """
    A tool is a pipeline or node that also has a name and description. When you add a Tool to an Agent, the Agent can
    invoke the underlying pipeline or node to answer questions. The wording of the description is important for the
    Agent to decide which tool is most useful for a given question.

    :param name: The name of the tool. The Agent uses this name to refer to the tool in the text the Agent generates.
        The name should be short, ideally one token, and a good description of what the tool can do, for example
        "Calculator" or "Search". Use only letters (a-z, A-Z), digits (0-9) and underscores (_).".
    :param pipeline_or_node: The pipeline or node to run when this tool is invoked by an Agent.
    :param description: A description of what the tool is useful for. An Agent can use this description for the decision
        when to use which tool. For example, a tool for calculations can be described by "useful for when you need to
        answer questions about math".
    """

    def __init__(
        self,
        name: str,
        pipeline_or_node: Union[
            BaseComponent,
            Pipeline,
            ExtractiveQAPipeline,
            DocumentSearchPipeline,
            GenerativeQAPipeline,
            SearchSummarizationPipeline,
            FAQPipeline,
            TranslationWrapperPipeline,
            RetrieverQuestionGenerationPipeline,
        ],
        description: str,
        output_variable: Optional[str] = "results",
    ):
        if re.search(r"\W", name):
            raise ValueError(
                f"Invalid name supplied for tool: '{name}'. Use only letters (a-z, A-Z), digits (0-9) and underscores (_)."
            )
        self.name = name
        self.pipeline_or_node = pipeline_or_node
        self.description = description
        self.output_variable = output_variable

    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        # We can only pass params to pipelines but not to nodes
        if isinstance(self.pipeline_or_node, (Pipeline, BaseStandardPipeline)):
            result = self.pipeline_or_node.run(query=tool_input, params=params)
        elif isinstance(self.pipeline_or_node, BaseRetriever):
            result = self.pipeline_or_node.run(query=tool_input, root_node="Query")
        else:
            result = self.pipeline_or_node.run(query=tool_input)
        return self._process_result(result)

    def _process_result(self, result: Any) -> str:
        # Base case: string or an empty container
        if not result or isinstance(result, str):
            return str(result)
        # Recursive case: process the result based on its type and return the result
        else:
            if isinstance(result, (tuple, list)):
                return self._process_result(result[0] if result else [])
            elif isinstance(result, dict):
                if self.output_variable not in result:
                    raise ValueError(
                        f"Tool {self.name} returned result {result} but "
                        f"output variable '{self.output_variable}' not found."
                    )
                return self._process_result(result[self.output_variable])
            elif isinstance(result, Answer):
                return self._process_result(result.answer)
            elif isinstance(result, Document):
                return self._process_result(result.content)
            else:
                return str(result)


class Agent:
    """
    An Agent answers queries by choosing between different tools, which are pipelines or nodes. It uses a large
    language model (LLM) to generate a thought based on the query, choose a tool, and generate the input for the
    tool. Based on the result returned by the tool the Agent can either stop if it now knows the answer or repeat the
    process of 1) generate thought, 2) choose tool, 3) generate input.

    Agents are useful for questions containing multiple subquestions that can be answered step-by-step (Multihop QA)
    using multiple pipelines and nodes as tools.
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Union[str, PromptTemplate] = "zero-shot-react",
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 8,
        tool_pattern: str = r'Tool:\s*(\w+)\s*Tool Input:\s*("?)([^"\n]+)\2\s*',
        final_answer_pattern: str = r"Final Answer:\s*(\w+)\s*",
    ):
        """
         Creates an Agent instance.

        :param prompt_node: The PromptNode that the Agent uses to decide which tool to use and what input to provide to it in each iteration.
        :param prompt_template: The name of a PromptTemplate supported by the PromptNode or a new PromptTemplate. It is used for generating thoughts and running Tools to answer queries step-by-step.
        :param tools: A List of Tools that the Agent can choose to run. If no Tools are provided, they need to be added with `add_tool()` before you can use the Agent.
        :param max_iterations: The number of times the Agent can run a tool plus then once infer it knows the final answer.
            Set at least to 2 so that the Agent can run one Tool and then infer it knows the final answer. Default 5.
        :param tool_pattern: A regular expression to extract the name of the Tool and the corresponding input from the text generated by the Agent.
        :param final_answer_pattern: A regular expression to extract final answer from the text generated by the Agent.
        """
        self.prompt_node = prompt_node
        self.prompt_template = (
            prompt_node.get_prompt_template(prompt_template) if isinstance(prompt_template, str) else prompt_template
        )
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.tool_names = ", ".join(self.tools.keys())
        self.tool_names_with_descriptions = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools.values()]
        )
        self.max_iterations = max_iterations
        self.tool_pattern = tool_pattern
        self.final_answer_pattern = final_answer_pattern
        send_custom_event(event=f"{type(self).__name__} initialized")

    def add_tool(self, tool: Tool):
        """
        Add the provided tool to the Agent and update the template for the Agent's PromptNode.

        :param tool: The Tool to add to the Agent. Any previously added tool with the same name will be overwritten.
        """
        self.tools[tool.name] = tool
        self.tool_names = ", ".join(self.tools.keys())
        self.tool_names_with_descriptions = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools.values()]
        )

    def has_tool(self, tool_name: str):
        """
        Check whether the Agent has a Tool registered under the provided tool name.

        :param tool_name: The name of the Tool for which to check whether the Agent has it.
        """
        return tool_name in self.tools

    def run(
        self, query: str, max_iterations: Optional[int] = None, params: Optional[dict] = None
    ) -> Dict[str, Union[str, List[Answer]]]:
        """
        Runs the Agent given a query and optional parameters to pass on to the tools used. The result is in the
        same format as a pipeline's result: a dictionary with a key `answers` containing a List of Answers

        :param query: The search query.
        :param max_iterations: The number of times the Agent can run a tool plus then once infer it knows the final answer.
            If set it should be at least 2 so that the Agent can run one tool and then infer it knows the final answer.
        :param params: A dictionary of parameters that you want to pass to those tools that are pipelines.
                       To pass a parameter to all nodes in those pipelines, use: `{"top_k": 10}`.
                       To pass a parameter to targeted nodes in those pipelines, use:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`.
                        Parameters can only be passed to tools that are pipelines but not nodes.
        """
        if not self.tools:
            raise AgentError(
                "Agents without tools cannot be run. Add at least one tool using `add_tool()` or set the parameter `tools` when initializing the Agent."
            )
        if max_iterations is None:
            max_iterations = self.max_iterations
        if max_iterations < 2:
            raise AgentError(
                f"max_iterations was set to {max_iterations} but it should be at least 2 so that the Agent can run one tool and then infer it knows the final answer."
            )

        agent_step = self._create_first_step(query)
        while not agent_step.is_terminal():
            agent_step = self._step(agent_step, params)

        return agent_step.final_answer(query=query)

    def _create_first_step(self, query):
        transcript = self._get_initial_transcript(query=query)
        return AgentStep(
            current_step=1,
            max_steps=self.max_iterations,
            final_answer_pattern=self.final_answer_pattern,
            llm_response="",  # no LLM response for the first step
            transcript=transcript,
        )

    def _step(self, current_step: AgentStep, params: Optional[dict] = None):
        preds = self.prompt_node(current_step.prepare_prompt())
        if not preds:
            raise AgentError(f"No output was generated by the Agent. Transcript:\n{current_step}")

        next_step = current_step.create_next_step(preds)
        observation = self._run_tool(next_step, params) if not next_step.is_terminal() else None
        next_step.completed(observation)
        return next_step

    def run_batch(
        self, queries: List[str], max_iterations: Optional[int] = None, params: Optional[dict] = None
    ) -> Dict[str, str]:
        """
        Runs the Agent in a batch mode

        :param queries: List of search queries.
        :param max_iterations: The number of times the Agent can run a tool plus then once infer it knows the final answer.
            If set it should be at least 2 so that the Agent can run one tool and then infer it knows the final answer.
        :param params: A dictionary of parameters that you want to pass to those tools that are pipelines.
                       To pass a parameter to all nodes in those pipelines, use: `{"top_k": 10}`.
                       To pass a parameter to targeted nodes in those pipelines, use:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`.
                        Parameters can only be passed to tools that are pipelines but not nodes.
        """
        results: Dict = {"queries": [], "answers": [], "transcripts": []}
        for query in queries:
            result = self.run(query=query, max_iterations=max_iterations, params=params)
            results["queries"].append(result["query"])
            results["answers"].append(result["answers"])
            results["transcripts"].append(result["transcript"])

        return results

    def _run_tool(self, next_step: AgentStep, params: Optional[Dict[str, Any]] = None) -> str:
        tool_name, tool_input = next_step.extract_tool_name_and_tool_input(self.tool_pattern)
        if tool_name is None or tool_input is None:
            raise AgentError(
                f"Could not identify the next tool or input for that tool from Agent's output. "
                f"Adjust the Agent's param 'tool_pattern' or 'prompt_template'. \n"
                f"# 'tool_pattern' to identify next tool: {self.tool_pattern} \n"
                f"# Agent Step:\n{next_step}"
            )
        if not self.has_tool(tool_name):
            raise AgentError(
                f"Cannot use the tool {tool_name} because it is not in the list of added tools {self.tools.keys()}."
                "Add the tool using `add_tool()` or include it in the parameter `tools` when initializing the Agent."
                f"Agent Step::\n{next_step}"
            )
        return self.tools[tool_name].run(tool_input, params)

    def _get_initial_transcript(self, query: str):
        """
        Fills the Agent's PromptTemplate with the query, tool names and descriptions

        :param query: The search query.
        """
        return next(
            self.prompt_template.fill(
                query=query, tool_names=self.tool_names, tool_names_with_descriptions=self.tool_names_with_descriptions
            ),
            "",
        )
