from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

from haystack import Pipeline, BaseComponent, Answer
from haystack.errors import AgentError
from haystack.nodes import PromptNode, PromptTemplate, BaseRetriever
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
    A tool is a pipeline or node that also has a name and description. When you add a Tool to an Agent, the Agent can
    invoke the underlying pipeline or node to answer questions. The wording of the description is important for the
    Agent to decide which tool is most useful for a given question.

    :param name: The name of the tool. The Agent uses this name to refer to the tool in the text the Agent generates.
        The name should consist of only one word/token and should be good description of what the tool can do,
        for example "Calculator" or "Search".
    :param pipeline_or_node: The pipeline or node to run when this tool is invoked by an Agent.
    :param description: A description of what the tool is useful for. An Agent can use this description for the decision
        when to use which tool. An exemplary description of a tool for calculations is "useful for when you need to
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
    ):
        self.name = name
        self.pipeline_or_node = pipeline_or_node
        self.description = description


class Agent:
    """
    An Agent answers queries by choosing between different tools, which are pipelines or nodes. It uses a large
    language model (LLM) to generate a thought based on the query, choose a tool, and generate the input for the
    tool. Based on the result returned by the tool the Agent can either stop if it knows the answer now or repeat the
    process of 1) generate thought, 2) choose tool, 3) generate input.

    Agents are useful for questions containing multiple subquestions that can be answered step-by-step (Multihop QA)
    using multiple pipelines and nodes as tools.

    :param prompt_node: The PromptNode that the Agent uses to decide which tool to use and what input to provide to it in each iteration.
    :param tools: A list of Tools that the Agent can choose to run. If no tools are provided, they need to be added with add_tool() before you can use the Agent.
    :param max_iterations: How many times the agent chooses another tool default 10. If set to 1, only one tool is chosen and there are no further iterations afterward.
    """

    def __init__(self, prompt_node: PromptNode, tools: Optional[List[Tool]] = None, max_iterations: int = 10):
        self.prompt_node = prompt_node
        if tools is not None:
            self.tools = {tool.name: tool for tool in tools}
            self.prompt_text = self._generate_prompt_text()
        else:
            self.tools = {}
            self.prompt_text = ""
        self.max_iterations = max_iterations
        send_custom_event(event=f"{type(self).__name__} initialized")

    def _generate_prompt_text(self) -> str:
        """
        Generate the initial prompt text for the Agent's PromptNode including descriptions and names of tools.
        """
        tool_names = ", ".join(self.tools.keys())
        tool_names_with_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools.values()])

        return (
            "You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
            "correctly, you have access to the following tools:\n\n"
            f"{tool_names_with_descriptions}\n\n"
            "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
            "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
            "for a final answer, respond with the `Final Answer:`\n\n"
            "Use the following format:\n\n"
            "Question: the question to be answered\n"
            "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
            f"Tool: [{tool_names}]\n"
            "Tool Input: the input for the tool\n"
            "Observation: the tool will respond with the result\n"
            "...\n"
            "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
            "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
            "---\n\n"
            "Question: $query\n"
            "Thought: Let's think step-by-step, I first need to "
        )

    def add_tool(self, tool: Tool):
        """
        Add a new tool to the Agent and update the template for the Agent's PromptNode.

        :param tool: The Tool to add to the Agent. Any previously added tool with the same name will be overwritten.
        """
        self.tools[tool.name] = tool
        self.prompt_text = self._generate_prompt_text()

    def run(self, query: str, params: Optional[dict] = None):
        """
        Runs the Agent given a query and optional parameters to pass on to the tools used. The result is in the
        same format as a pipeline's result: a dictionary with a key "answers" containing a List of Answers

        :param query: The search query (for query pipelines only).
        :param params: A dictionary of parameters that you want to pass to those tools that are pipelines.
                       To pass a parameter to all nodes in those pipelines, use: `{"top_k": 10}`.
                       To pass a parameter to targeted nodes in those pipelines, use:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`.
                        Parameters can only be passed to tools that are pipelines but not nodes.
        """
        if len(self.tools) == 0:
            raise AgentError("Agents without tools cannot be run. Add at least one tool using add_tool().")
        if params and "max_iterations" in params:
            remaining_iterations = params["max_iterations"]
        else:
            remaining_iterations = self.max_iterations

        transcript = self.prompt_text
        final_answer = ""

        # In each iteration, choose a tool, generate input for it, and run it until the final answer is found or the
        # maximum number of iterations is reached.
        while remaining_iterations > 0:
            remaining_iterations -= 1

            prompt_template = PromptTemplate("think-step-by-step", transcript, ["query"])
            pred = self.prompt_node.prompt(prompt_template=prompt_template, query=query)
            transcript += f"{pred[0]}\n"

            tool_name, tool_input = self._parse_tool_name_and_tool_input(pred=pred[0])
            if tool_name == "Final Answer":
                final_answer = tool_input
                break
            if tool_name not in self.tools:
                # TODO we could try to recover from this error by rephrasing the question and retrying
                raise AgentError(
                    f'The Agent tried to use a tool "{tool_name}" but the registered tools are only {self.tools.keys()}.'
                )
            pipeline_or_node = self.tools[tool_name].pipeline_or_node
            # We can only pass params to pipelines but not to nodes
            if isinstance(pipeline_or_node, (Pipeline, BaseStandardPipeline)):
                result = pipeline_or_node.run(query=tool_input, params=params)
            else:
                if isinstance(pipeline_or_node, BaseRetriever):
                    result = pipeline_or_node.run(query=tool_input, root_node="Query")
                else:
                    result = pipeline_or_node.run(query=tool_input)

            observation = ""
            # if pipeline_or_node is a node it returns a tuple. We use only the output but not the name of the output.
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, dict):
                if "results" in result and len(result["results"]) > 0:
                    observation = result["results"][0]
                elif "answers" in result and len(result["answers"]) > 0:
                    observation = result["answers"][0].answer
                elif "documents" in result and len(result["documents"]) > 0:
                    observation = result["documents"][0].content
                else:
                    # no answer/document/result was returned
                    observation = ""
            transcript += f"Observation: {observation}\nThought: Now that I know that {tool_input} is {observation}, I "

        else:
            logger.warning(
                "Maximum number of agent iterations (%s) reached for query (%s). Increase max_iterations "
                "or no answer can be provided for this query.",
                self.max_iterations,
                query,
            )

        return {"query": query, "answers": [Answer(answer=final_answer, type="generative")], "transcript": transcript}

    def run_batch(self, queries: List[str], params: Optional[dict] = None):
        """
        Runs the Agent in a batch mode

        :param queries: List of search queries (for query pipelines only).
        :param params: A dictionary of parameters that you want to pass to the tools.
                       To pass a parameter to all tools, use: `{"top_k": 10}`.
                       To pass a parameter to targeted tools, run:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`
        """
        results: Dict = {"queries": [], "answers": [], "transcripts": []}
        for query in queries:
            result = self.run(query=query, params=params)
            results["queries"].append(result["query"])
            results["answers"].append(result["answers"])
            results["transcripts"].append(result["transcript"])

        return results

    def _parse_tool_name_and_tool_input(self, pred: str) -> Tuple[str, str]:
        """
        Parse the tool name and the tool input from the prediction output of the Agent's PromptNode.

        :param pred: Prediction output of the Agent's PromptNode from which to parse the tool and tool input
        """
        final_answer_string = "Final Answer: "
        tool_name_string = "Tool: "
        tool_input_string = "Tool Input: "

        lines = [line for line in pred.split("\n") if line]
        if len(lines) == 0:
            # TODO we could try to recover from this error by rephrasing the question and retrying
            raise AgentError("Wrong output format. The output does not contain multiple lines.")
        if lines[-1].startswith(final_answer_string):
            directive = lines[-1][len(final_answer_string) :]
            return "Final Answer", directive
        if not lines[-1].startswith(tool_input_string):
            # TODO we could try to recover from this error by rephrasing the question and retrying
            raise AgentError(f'Wrong output format. The last line does not start with "{tool_input_string}"')
        if not lines[-2].startswith(tool_name_string):
            # TODO we could try to recover from this error by rephrasing the question and retrying
            raise AgentError(f'Wrong output format. The second to last line does not start with "{tool_name_string}"')
        tool_name = lines[-2][len(tool_name_string) :]
        tool_input = lines[-1][len(tool_input_string) :]
        return tool_name.strip('" []').strip(), tool_input.strip('" ')

    @classmethod
    def load_from_yaml(cls, path: Path, agent_name: Optional[str] = None, strict_version_check: bool = False):
        """
        Load Agent from a YAML file defining the individual tools. A single YAML can declare multiple agents, in which case an explicit `agent_name` must
        be passed.

        Here's a sample configuration:
        ...
        """
        raise NotImplementedError

    def save_to_yaml(self, path: Path, return_defaults: bool = False):
        """
        Save a YAML configuration for the Agent that can be used with `Agent.load_from_yaml()`.

        :param path: path of the output YAML file.
        :param return_defaults: whether to output parameters that have the default values.
        """
        # config = self.get_config(return_defaults=return_defaults)
        # with open(path, "w") as outfile:
        #     yaml.dump("", outfile, default_flow_style=False)
        raise NotImplementedError

    def send_agent_event(self):
        raise NotImplementedError
