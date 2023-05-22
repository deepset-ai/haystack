from typing import Optional, Callable, Union

from haystack.agents import Agent
from haystack.agents.base import ToolsManager, react_parameter_resolver
from haystack.agents.memory import Memory, ConversationMemory, ConversationSummaryMemory
from haystack.nodes import PromptNode, PromptTemplate


class ConversationalAgent(Agent):
    """
    A conversational agent that can be used to build a conversational chat applications.

    Here is an example of how you can create a simple chat application:
    ```
    import os

    from haystack.agents.base import ConversationalAgent
    from haystack.agents.memory import ConversationSummaryMemory
    from haystack.nodes import PromptNode

    pn = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)
    agent = ConversationalAgent(pn, memory=ConversationSummaryMemory(pn))

    while True:
        user_input = input("Human (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        elif user_input.lower() == "memory":
            print("\nMemory:\n", agent.memory.load())
        else:
            assistant_response = agent.run(user_input)
            print("\nAssistant:", assistant_response)

    ```
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        memory: Optional[Memory] = None,
        prompt_parameters_resolver: Optional[Callable] = None,
    ):
        """
        Creates a new ConversationalAgent instance

        :param prompt_node: A PromptNode used to communicate with LLM.
        :param prompt_template: A string or PromptTemplate instance to use as the prompt template. If no prompt_template
        is provided, the agent will use the "conversational-agent" template.
        :param memory: A memory instance for storing conversation history and other relevant data, defaults to
        ConversationMemory.
        :param prompt_parameters_resolver: An optional callable for resolving prompt template parameters,
        defaults to a callable that returns a dictionary with the query and the conversation history.
        """
        super().__init__(
            prompt_node=prompt_node,
            prompt_template=prompt_template or "conversational-agent",
            max_steps=2,
            memory=memory if memory else ConversationMemory(),
            prompt_parameters_resolver=prompt_parameters_resolver
            if prompt_parameters_resolver
            else lambda query, agent, **kwargs: {"query": query, "history": agent.memory.load()},
            final_answer_pattern=r"^([\s\S]+)$",
        )


class ConversationalAgentWithTools(Agent):
    """
    A ConversationalAgentWithTools is an extension of the Agent class that supports the use of tools in
    conversational chat applications. This agent can make use of the ToolsManager to manage a set of tools
    and seamlessly integrate them into the conversation.

    Example usage:

    ```
    import os

    from haystack.agents.base import ConversationalAgentWithTools
    from haystack.agents.memory import ConversationSummaryMemory
    from haystack.nodes import PromptNode
    from haystack.tools import ToolsManager, Tool

    # Initialize a PromptNode and a ToolsManager with the desired tools
    pn = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)
    tools_manager = ToolsManager([Tool(name="ExampleTool", pipeline_or_node=example_tool_node)])

    # Create the ConversationalAgentWithTools instance
    agent = ConversationalAgentWithTools(pn, tools_manager=tools_manager)

    # Use the agent in a chat application
    while True:
        user_input = input("Human (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        else:
            assistant_response = agent.run(user_input)
            print("\nAssistant:", assistant_response)
    ```
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        tools_manager: Optional[ToolsManager] = None,
        max_steps: int = 5,
        memory: Optional[Memory] = None,
        prompt_parameters_resolver: Optional[Callable] = None,
        final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
    ):
        """
        Creates an instance of ConversationalAgentWithTools.

        :param prompt_node: A PromptNode used to communicate with LLM.
        :param prompt_template: A string or PromptTemplate instance to use as the prompt template. If no prompt_template
        is provided, the agent will use the "conversational-agent-with-tools" template.
        :param tools_manager: A ToolsManager instance to manage tools for the agent. If no tools_manager is provided,
        the agent will install an empty ToolsManager instance.
        :param max_steps: The maximum number of steps for the agent to take, defaults to 5.
        :param memory: A memory object for storing conversation history and other relevant data, defaults to
        ConversationSummaryMemory if no memory is provided.
        :param prompt_parameters_resolver: An optional callable for resolving prompt template parameters,
        defaults to keys: query, tool_names, tool_names_with_descriptions, transcript. Their values are set appropriately.
        :param final_answer_pattern: A regular expression to extract the final answer from the text the Agent generated.
        """
        super().__init__(
            prompt_node=prompt_node,
            prompt_template=prompt_template or "conversational-agent-with-tools",
            tools_manager=tools_manager,
            max_steps=max_steps,
            memory=memory if memory else ConversationSummaryMemory(prompt_node),
            prompt_parameters_resolver=prompt_parameters_resolver
            if prompt_parameters_resolver
            else react_parameter_resolver,
            final_answer_pattern=final_answer_pattern,
        )
