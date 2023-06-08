from typing import Optional, Dict, Any, List
import logging

from haystack.errors import AgentError
from haystack.agents.base import Tool, ToolsManager
from haystack.agents import Agent, AgentStep
from haystack.agents.memory import Memory, ConversationMemory
from haystack.nodes import PromptNode

logger = logging.getLogger(__name__)


def agent_without_tools_parameter_resolver(query: str, agent: Agent, **kwargs) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct based agents without tools that returns the query, the history.
    """
    return {"query": query, "history": agent.memory.load()}


def conversational_agent_parameter_resolver(
    query: str, agent: Agent, agent_step: AgentStep, **kwargs
) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct based agents that returns the query, the tool names, the tool names
    with descriptions, the history of the conversation, and the transcript (internal monologue).
    """
    return {
        "query": query,
        "tool_names": agent.tm.get_tool_names(),
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "history": agent.memory.load(),
    }


class ConversationalAgent(Agent):
    """
    A ConversationalAgent is an extension of the Agent class that supports the use of tools in
    conversational chat applications. This agent can make use of the ToolsManager to manage a set of tools and seamlessly integrate them into the conversation.

    If no tools are provided, the Agent will be initialized with necessary arguments to have a basic chat application.

    Here is an example of how you can create a chat application with tools:
    ```
    import os

    from haystack.agents.conversational import ConversationalAgent
    from haystack.nodes import PromptNode
    from haystack.tools import ToolsManager, Tool

    # Initialize a PromptNode and a ToolsManager with the desired tools
    prompt_node = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)
    tools_manager = ToolsManager([Tool(name="ExampleTool", pipeline_or_node=example_tool_node)])

    # Create the ConversationalAgent instance
    agent = ConversationalAgent(prompt_node, tools_manager=tools_manager)

    # Use the agent in a chat application
    while True:
        user_input = input("Human (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        else:
            assistant_response = agent.run(user_input)
            print("\nAssistant:", assistant_response)
    ```

    If you want to have any tools, you can create a ConversationalAgent only with a PromptNode:
    ```
    import os

    from haystack.agents.conversational import ConversationalAgent
    from haystack.nodes import PromptNode

    # Initialize a PromptNode
    prompt_node = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)

    # Create the ConversationalAgent instance
    agent = ConversationalAgent(prompt_node)
    ```
    """

    def __init__(self, prompt_node: PromptNode, tools: Optional[List[Tool]] = None, memory: Optional[Memory] = None):
        """
        Creates a new ConversationalAgent instance

        :param prompt_node: A PromptNode that the Agent uses to decide which tool to use and what input to provide to
        it in each iteration. If no tools are provided, model provided with PromptNode will be used to chat with.
        :param tools: A list of tools to use in the Agent. Each tool must have a unique name.
        :param memory: A memory object for storing conversation history and other relevant data, defaults to
        ConversationMemory if no memory is provided.
        """

        if tools:
            super().__init__(
                prompt_node=prompt_node,
                memory=memory if memory else ConversationMemory(),
                tools_manager=ToolsManager(tools=tools),
                max_steps=5,
                prompt_template="conversational-agent",
                final_answer_pattern=r"Final Answer\s*:\s*(.*)",
                prompt_parameters_resolver=conversational_agent_parameter_resolver,
            )
        else:
            logger.warning("ConversationalAgent is created without tools")

            super().__init__(
                prompt_node=prompt_node,
                memory=memory if memory else ConversationMemory(),
                max_steps=2,
                prompt_template="conversational-agent-without-tools",
                final_answer_pattern=r"^([\s\S]+)$",
                prompt_parameters_resolver=agent_without_tools_parameter_resolver,
            )

    def add_tool(self, tool: Tool):
        if len(self.tm.tools) == 0:
            raise AgentError(
                "You cannot add tools after initializing the ConversationalAgent without any tools. If you want to add tools, reinitailize the ConversationalAgent and provide `tools`."
            )
        return super().add_tool(tool)
