from typing import Optional, List, Union
import logging

from haystack.errors import AgentError
from haystack.agents.base import Tool, ToolsManager, Agent
from haystack.agents.memory import Memory, ConversationMemory
from haystack.nodes import PromptNode, PromptTemplate
from haystack.agents.utils import conversational_agent_parameter_resolver, agent_without_tools_parameter_resolver

logger = logging.getLogger(__name__)


class ConversationalAgent(Agent):
    """
    A ConversationalAgent is an extension of the Agent class that enables the use of tools with several default parameters.
    ConversationalAgent can manage a set of tools and seamlessly integrate them into the conversation.
    If no tools are provided, the agent will be initialized to have a basic chat application.

    Here is an example how you can create a chat application with tools:

    ```python
    import os

    from haystack.agents.conversational import ConversationalAgent
    from haystack.nodes import PromptNode
    from haystack.agents.base import Tool

    # Initialize a PromptNode and the desired tools
    prompt_node = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)
    tools = [Tool(name="ExampleTool", pipeline_or_node=example_tool_node)]

    # Create the ConversationalAgent instance
    agent = ConversationalAgent(prompt_node, tools=tools)

    # Use the agent in a chat application
    while True:
        user_input = input("Human (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        else:
            assistant_response = agent.run(user_input)
            print("Assistant:", assistant_response)
    ```

    If you don't want to have any tools in your chat app, you can create a ConversationalAgent only with a PromptNode:

    ```python
    import os

    from haystack.agents.conversational import ConversationalAgent
    from haystack.nodes import PromptNode

    # Initialize a PromptNode
    prompt_node = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)

    # Create the ConversationalAgent instance
    agent = ConversationalAgent(prompt_node)
    ```

    If you're looking for more customization, check out [Agent](https://docs.haystack.deepset.ai/reference/agent-api).
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Creates a new ConversationalAgent instance.

        :param prompt_node: A PromptNode used by Agent to decide which tool to use and what input to provide to it
        in each iteration. If there are no tools added, the model specified with PromptNode will be used for chatting.
        :param prompt_template: A new PromptTemplate or the name of an existing PromptTemplate for the PromptNode. It's
        used for keeping the chat history, generating thoughts and choosing tools (if provided) to answer queries. It defaults to
        to "conversational-agent" if there is at least one tool provided and "conversational-agent-without-tools" otherwise.
        :param tools: A list of tools to use in the Agent. Each tool must have a unique name.
        :param memory: A memory object for storing conversation history and other relevant data, defaults to
        ConversationMemory if no memory is provided.
        :param max_steps: The number of times the Agent can run a tool +1 to let it infer it knows the final answer. It defaults to 5 if there is at least one tool provided and 2 otherwise.
        """

        if tools:
            super().__init__(
                prompt_node=prompt_node,
                memory=memory if memory else ConversationMemory(),
                tools_manager=ToolsManager(tools=tools),
                max_steps=max_steps if max_steps else 5,
                prompt_template=prompt_template if prompt_template else "conversational-agent",
                final_answer_pattern=r"Final Answer\s*:\s*(.*)",
                prompt_parameters_resolver=conversational_agent_parameter_resolver,
            )
        else:
            logger.warning("ConversationalAgent is created without tools")

            super().__init__(
                prompt_node=prompt_node,
                memory=memory if memory else ConversationMemory(),
                max_steps=max_steps if max_steps else 2,
                prompt_template=prompt_template if prompt_template else "conversational-agent-without-tools",
                final_answer_pattern=r"^([\s\S]+)$",
                prompt_parameters_resolver=agent_without_tools_parameter_resolver,
            )

    def add_tool(self, tool: Tool):
        if len(self.tm.tools) == 0:
            raise AgentError(
                "You cannot add tools after initializing the ConversationalAgent without any tools. If you want to add tools, reinitailize the ConversationalAgent and provide `tools`."
            )
        return super().add_tool(tool)
