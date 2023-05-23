from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Callable
from hashlib import md5
from typing import List, Optional, Union, Dict, Any, Tuple

from events import Events

from haystack import Pipeline, BaseComponent, Answer, Document
from haystack.agents.memory import Memory, NoMemory
from haystack.telemetry import send_event
from haystack.agents.agent_step import AgentStep
from haystack.agents.types import Color, AgentTokenStreamingHandler
from haystack.agents.utils import print_text
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
    WebQAPipeline,
)

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
            WebQAPipeline,
        ],
        description: str,
        output_variable: str = "results",
        logging_color: Color = Color.YELLOW,
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
        self.logging_color = logging_color

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


class ToolsManager:
    """
    The ToolsManager manages tools for an Agent.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        tool_pattern: str = r"Tool:\s*(\w+)\s*Tool Input:\s*(?:\"([\s\S]*?)\"|((?:.|\n)*))\s*",
    ):
        """
        :param tools: A list of tools to add to the ToolManager. Each tool must have a unique name.
        :param tool_pattern: A regular expression pattern that matches the text that the Agent generates to invoke
            a tool.
        """
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools} if tools else {}
        self.tool_pattern = tool_pattern
        self.callback_manager = Events(("on_tool_start", "on_tool_finish", "on_tool_error"))

    def add_tool(self, tool: Tool):
        """
        Add a tool to the Agent. This also updates the PromptTemplate for the Agent's PromptNode with the tool name.

        :param tool: The tool to add to the Agent. Any previously added tool with the same name will be overwritten.
        Example:
        `agent.add_tool(
            Tool(
                name="Calculator",
                pipeline_or_node=calculator
                description="Useful when you need to answer questions about math."
            )
        )
        """
        self.tools[tool.name] = tool

    @property
    def tools(self):
        return self._tools

    def get_tool_names(self) -> str:
        """
        Returns a string with the names of all registered tools.
        """
        return ", ".join(self.tools.keys())

    def get_tools(self) -> List[Tool]:
        """
        Returns a list of all registered tool instances.
        """
        return list(self.tools.values())

    def get_tool_names_with_descriptions(self) -> str:
        """
        Returns a string with the names and descriptions of all registered tools.
        """
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools.values()])

    def run_tool(self, llm_response: str, params: Optional[Dict[str, Any]] = None) -> str:
        tool_result: str = ""
        if self.tools:
            tool_name, tool_input = self.extract_tool_name_and_tool_input(llm_response)
            if tool_name and tool_input:
                tool: Tool = self.tools[tool_name]
                try:
                    self.callback_manager.on_tool_start(tool_input, tool=tool)
                    tool_result = tool.run(tool_input, params)
                    self.callback_manager.on_tool_finish(
                        tool_result,
                        observation_prefix="Observation: ",
                        llm_prefix="Thought: ",
                        color=tool.logging_color,
                    )
                except Exception as e:
                    self.callback_manager.on_tool_error(e, tool=self.tools[tool_name])
                    raise e
        return tool_result

    def extract_tool_name_and_tool_input(self, llm_response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the tool name and the tool input from the PromptNode response.
        :param llm_response: The PromptNode response.
        :return: A tuple containing the tool name and the tool input.
        """
        tool_match = re.search(self.tool_pattern, llm_response)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_input = tool_match.group(2) or tool_match.group(3)
            return tool_name.strip('" []\n').strip(), tool_input.strip('" \n')
        return None, None


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
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        tools_manager: Optional[ToolsManager] = None,
        memory: Optional[Memory] = None,
        prompt_parameters_resolver: Optional[Callable] = None,
        max_steps: int = 8,
        final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
        streaming: bool = True,
    ):
        """
         Creates an Agent instance.

        :param prompt_node: The PromptNode that the Agent uses to decide which tool to use and what input to provide to
        it in each iteration.
        :param prompt_template: The name of a PromptTemplate for the PromptNode. It's used for generating thoughts and
        choosing tools to answer queries step-by-step. You can use the default `zero-shot-react` template or create a
        new template in a similar format.
        with `add_tool()` before running the Agent.
        :param tools_manager: A ToolsManager instance that the Agent uses to run tools. Each tool must have a unique name.
        You can also add tools with `add_tool()` before running the Agent.
        :param memory: A Memory instance that the Agent uses to store information between iterations.
        :param prompt_parameters_resolver: A callable that takes query, agent, and agent_step as parameters and returns
        a dictionary of parameters to pass to the prompt_template. The default is a callable that returns a dictionary
        of keys and values needed for the React agent prompt template.
        :param max_steps: The number of times the Agent can run a tool +1 to let it infer it knows the final answer.
            Set it to at least 2, so that the Agent can run one a tool once and then infer it knows the final answer.
            The default is 8.
        :param final_answer_pattern: A regular expression to extract the final answer from the text the Agent generated.
        :param streaming: Whether to use streaming or not. If True, the Agent will stream response tokens from the LLM.
        If False, the Agent will wait for the LLM to finish generating the response and then process it. The default is
        True.
        """
        self.max_steps = max_steps
        self.tm = tools_manager or ToolsManager()
        self.memory = memory or NoMemory()
        self.callback_manager = Events(("on_agent_start", "on_agent_step", "on_agent_finish", "on_new_token"))
        self.prompt_node = prompt_node
        prompt_template = prompt_template or "zero-shot-react"
        resolved_prompt_template = prompt_node.get_prompt_template(prompt_template)
        if not resolved_prompt_template:
            raise ValueError(
                f"Prompt template '{prompt_template}' not found. Please check the spelling of the template name."
            )
        self.prompt_template = resolved_prompt_template
        react_parameter_resolver: Callable[
            [str, Agent, AgentStep, Dict[str, Any]], Dict[str, Any]
        ] = lambda query, agent, agent_step, **kwargs: {
            "query": query,
            "tool_names": agent.tm.get_tool_names(),
            "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
            "transcript": agent_step.transcript,
        }
        self.prompt_parameters_resolver = (
            prompt_parameters_resolver if prompt_parameters_resolver else react_parameter_resolver
        )
        self.final_answer_pattern = final_answer_pattern
        self.add_default_logging_callbacks(streaming=streaming)
        self.hash = None
        self.last_hash = None
        self.update_hash()

    def update_hash(self):
        """
        Used for telemetry. Hashes the tool classnames to send an event only when they change.
        See haystack/telemetry.py::send_event
        """
        try:
            tool_names = " ".join([tool.pipeline_or_node.__class__.__name__ for tool in self.tm.get_tools()])
            self.hash = md5(tool_names.encode()).hexdigest()
        except Exception as exc:
            logger.debug("Telemetry exception: %s", str(exc))
            self.hash = "[an exception occurred during hashing]"

    def add_default_logging_callbacks(self, agent_color: Color = Color.GREEN, streaming: bool = False) -> None:
        def on_tool_finish(
            tool_output: str,
            color: Optional[Color] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            print_text(observation_prefix)  # type: ignore
            print_text(tool_output, color=color)
            print_text(f"\n{llm_prefix}")

        def on_agent_start(**kwargs: Any) -> None:
            agent_name = kwargs.pop("name", "react")
            print_text(f"\nAgent {agent_name} started with {kwargs}\n")

        self.tm.callback_manager.on_tool_finish += on_tool_finish
        self.callback_manager.on_agent_start += on_agent_start

        if streaming:
            self.callback_manager.on_new_token += lambda token, **kwargs: print_text(token, color=agent_color)
        else:
            self.callback_manager.on_agent_step += lambda agent_step: print_text(
                agent_step.prompt_node_response, end="\n", color=agent_color
            )

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
        self.tm.add_tool(tool)

    def has_tool(self, tool_name: str) -> bool:
        """
        Check whether the Agent has a tool with the name you provide.

        :param tool_name: The name of the tool for which you want to check whether the Agent has it.
        """
        return tool_name in self.tm.tools

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
        try:
            if not self.hash == self.last_hash:
                self.last_hash = self.hash
                send_event(event_name="Agent", event_properties={"llm.agent_hash": self.hash})
        except Exception as exc:
            logger.debug("Telemetry exception: %s", exc)

        self.callback_manager.on_agent_start(name=self.prompt_template.name, query=query, params=params)
        agent_step = self.create_agent_step(max_steps)
        try:
            while not agent_step.is_last():
                agent_step = self._step(query, agent_step, params)
        finally:
            self.callback_manager.on_agent_finish(agent_step)
        return agent_step.final_answer(query=query)

    def _step(self, query: str, current_step: AgentStep, params: Optional[dict] = None):
        # plan next step using the LLM
        prompt_node_response = self._plan(query, current_step)

        # from the LLM response, create the next step
        next_step = current_step.create_next_step(prompt_node_response)
        self.callback_manager.on_agent_step(next_step)

        # run the tool selected by the LLM
        observation = self.tm.run_tool(next_step.prompt_node_response, params) if not next_step.is_last() else None

        # save the input, output and observation to memory (if memory is enabled)
        memory_data = self.prepare_data_for_memory(input=query, output=prompt_node_response, observation=observation)
        self.memory.save(data=memory_data)

        # update the next step with the observation
        next_step.completed(observation)
        return next_step

    def _plan(self, query, current_step):
        # first resolve prompt template params
        template_params = self.prompt_parameters_resolver(query=query, agent=self, agent_step=current_step)

        # if prompt node has no default prompt template, use agent's prompt template
        if self.prompt_node.default_prompt_template is None:
            prepared_prompt = next(self.prompt_template.fill(**template_params))
            prompt_node_response = self.prompt_node(
                prepared_prompt, stream_handler=AgentTokenStreamingHandler(self.callback_manager)
            )
        # otherwise, if prompt node has default prompt template, use it
        else:
            prompt_node_response = self.prompt_node(
                stream_handler=AgentTokenStreamingHandler(self.callback_manager), **template_params
            )
        return prompt_node_response

    def create_agent_step(self, max_steps: Optional[int] = None) -> AgentStep:
        """
        Create an AgentStep object. Override this method to customize the AgentStep class used by the Agent.
        """
        return AgentStep(max_steps=max_steps or self.max_steps, final_answer_pattern=self.final_answer_pattern)

    def prepare_data_for_memory(self, **kwargs) -> dict:
        """
        Prepare data for saving to the Agent's memory. Override this method to customize the data saved to the memory.
        """
        return {
            k: v if isinstance(v, str) else next(iter(v)) for k, v in kwargs.items() if isinstance(v, (str, Iterable))
        }
