from __future__ import annotations

import logging
import re
from typing import List, Optional, Union, Dict, Any

from haystack import Pipeline, BaseComponent, Answer, Document
from haystack.agents.agent_step import AgentStep
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


class Tool:
    """
    Agent uses tools to find the best answer. A tool is a pipeline or a node. When you add a tool to an Agent, the Agent
    can invoke the underlying pipeline or node to answer questions.

    You must provide a name and a description for each tool. The name should be short and should indicate what the tool
    can do. The description should explain what the tool is useful for. The Agent uses the description to decide when
    to use a tool, so the wording you use is important.

    :param name: The name of the tool. The Agent uses this name to refer to the tool in the text the Agent generates.
        The name should be short, ideally one token, and a good description of what the tool can do, for example:
        "Calculator" or "Search". Use only letters (a-z, A-Z), digits (0-9) and underscores (_)."
    :param pipeline_or_node: The pipeline or node to run when the Agent invokes this tool.
    :param description: A description of what the tool is useful for. The Agent uses this description to decide
        when to use which tool. For example, you can describe a tool for calculations by "useful for when you need to

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
                f"Invalid name supplied for tool: '{name}'. Use only letters (a-z, A-Z), digits (0-9) and "
                f"underscores (_)."
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
    An Agent answers queries using the tools you give to it. The tools are pipelines or nodes. The Agent uses a large
    language model (LLM) through the PromptNode you initialize it with. To answer a query, the Agent follows this
    sequence:

    1. It generates a thought based on the query.
    2. It decides which tool to use.
    3. It generates the input for the tool.
    4. Based on the output it gets from the tool, the Agent can either stop if it now knows the answer or repeat the
    process of 1) generate thought, 2) choose tool, 3) generate input.

    Agents are useful for questions containing multiple sub questions that can be answered step-by-step (Multi-hop QA)
    using multiple pipelines and nodes as tools.
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Union[str, PromptTemplate] = "zero-shot-react",
        tools: Optional[List[Tool]] = None,
        max_steps: int = 8,
        tool_pattern: str = r'Tool:\s*(\w+)\s*Tool Input:\s*("?)([^"\n]+)\2\s*',
        final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
    ):
        """
         Creates an Agent instance.

        :param prompt_node: The PromptNode that the Agent uses to decide which tool to use and what input to provide to
        it in each iteration.
        :param prompt_template: The name of a PromptTemplate for the PromptNode. It's used for generating thoughts and
        choosing tools to answer queries step-by-step. You can use the default `zero-shot-react` template or create a
        new template in a similar format.
        :param tools: A list of tools the Agent can run. If you don't specify any tools here, you must add them
        with `add_tool()` before running the Agent.
        :param max_steps: The number of times the Agent can run a tool +1 to let it infer it knows the final answer.
            Set it to at least 2, so that the Agent can run one a tool once and then infer it knows the final answer.
            The default is 5.
        :param tool_pattern: A regular expression to extract the name of the tool and the corresponding input from the
        text the Agent generated.
        :param final_answer_pattern: A regular expression to extract the final answer from the text the Agent generated.
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
        self.max_steps = max_steps
        self.tool_pattern = tool_pattern
        self.final_answer_pattern = final_answer_pattern
        send_custom_event(event=f"{type(self).__name__} initialized")

    def add_tool(self, tool: Tool):
        """
        Add a tool to the Agent. This also updates the PromptTemplate for the Agent's PromptNode with the tool name.

        :param tool: The tool to add to the Agent. Any previously added tool with the same name will be overwritten.
        Example:
        `agent.add_tool(
            Tool(
                name="Calculator",
                pipeline_or_node=calculator
                description="Useful when you need to answer questions about math"
            )
        )
        """
        self.tools[tool.name] = tool
        self.tool_names = ", ".join(self.tools.keys())
        self.tool_names_with_descriptions = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools.values()]
        )

    def has_tool(self, tool_name: str):
        """
        Check whether the Agent has a tool with the name you provide.

        :param tool_name: The name of the tool for which you want to check whether the Agent has it.
        """
        return tool_name in self.tools

    def run(
        self, query: str, max_steps: Optional[int] = None, params: Optional[dict] = None
    ) -> Dict[str, Union[str, List[Answer]]]:
        """
        Runs the Agent given a query and optional parameters to pass on to the tools used. The result is in the
        same format as a pipeline's result: a dictionary with a key `answers` containing a list of answers.

        :param query: The search query
        :param max_steps: The number of times the Agent can run a tool +1 to infer it knows the final answer.
            If you want to set it, make it at least 2 so that the Agent can run a tool once and then infer it knows the
            final answer.
        :param params: A dictionary of parameters you want to pass to the tools that are pipelines.
                       To pass a parameter to all nodes in those pipelines, use the format: `{"top_k": 10}`.
                       To pass a parameter to targeted nodes in those pipelines, use the format:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`.
                        You can only pass parameters to tools that are pipelines, but not nodes.
        """
        if not self.tools:
            raise AgentError(
                "An Agent needs tools to run. Add at least one tool using `add_tool()` or set the parameter `tools` "
                "when initializing the Agent."
            )
        if max_steps is None:
            max_steps = self.max_steps
        if max_steps < 2:
            raise AgentError(
                f"max_steps must be at least 2 to let the Agent use a tool once and then infer it knows the final "
                f"answer. It was set to {max_steps}."
            )

        agent_step = self._create_first_step(query, max_steps)
        while not agent_step.is_last():
            agent_step = self._step(agent_step, params)

        return agent_step.final_answer(query=query)

    def _create_first_step(self, query: str, max_steps: int = 10):
        transcript = self._get_initial_transcript(query=query)
        return AgentStep(
            current_step=1,
            max_steps=max_steps,
            final_answer_pattern=self.final_answer_pattern,
            prompt_node_response="",  # no LLM response for the first step
            transcript=transcript,
        )

    def _step(self, current_step: AgentStep, params: Optional[dict] = None):
        # plan next step using the LLM
        prompt_node_response = self.prompt_node(current_step.prepare_prompt())

        # from the LLM response, create the next step
        next_step = current_step.create_next_step(prompt_node_response)

        # run the tool selected by the LLM
        observation = self._run_tool(next_step, params) if not next_step.is_last() else None

        # update the next step with the observation
        next_step.completed(observation)
        return next_step

    def run_batch(
        self, queries: List[str], max_steps: Optional[int] = None, params: Optional[dict] = None
    ) -> Dict[str, str]:
        """
        Runs the Agent in a batch mode.

        :param queries: List of search queries.
        :param max_steps: The number of times the Agent can run a tool +1 to infer it knows the final answer.
            If you want to set it, make it at least 2 so that the Agent can run a tool once and then infer it knows
            the final answer.
        :param params: A dictionary of parameters you want to pass to the tools that are pipelines.
                       To pass a parameter to all nodes in those pipelines, use the format: `{"top_k": 10}`.
                       To pass a parameter to targeted nodes in those pipelines, use the format:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`.
                        You can only pass parameters to tools that are pipelines but not nodes.
        """
        results: Dict = {"queries": [], "answers": [], "transcripts": []}
        for query in queries:
            result = self.run(query=query, max_steps=max_steps, params=params)
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
                f"The tool {tool_name} wasn't added to the Agent tools: {self.tools.keys()}."
                "Add the tool using `add_tool()` or include it in the parameter `tools` when initializing the Agent."
                f"Agent Step::\n{next_step}"
            )
        return self.tools[tool_name].run(tool_input, params)

    def _get_initial_transcript(self, query: str):
        """
        Fills the Agent's PromptTemplate with the query, tool names, and descriptions.

        :param query: The search query.
        """
        return next(
            self.prompt_template.fill(
                query=query, tool_names=self.tool_names, tool_names_with_descriptions=self.tool_names_with_descriptions
            ),
            "",
        )
