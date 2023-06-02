from typing import Optional, Callable, Union

from haystack.agents.base import ToolsManager, react_parameter_resolver
from haystack.agents import Agent
from haystack.agents.memory import Memory, ConversationMemory
from haystack.nodes import PromptNode, PromptTemplate


class ConversationalAgent(Agent):
    """
    A ConversationalAgent is an extension of the Agent class that supports the use of tools in
    conversational chat applications. This agent can make use of the ToolsManager to manage a set of tools and seamlessly integrate them into the conversation.

    Here is an example of how you can create a simple chat application:
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
    """

    def __init__(
        self,
        prompt_node: PromptNode,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        tools_manager: Optional[ToolsManager] = None,
        memory: Optional[Memory] = ConversationMemory(),
        prompt_parameters_resolver: Optional[Callable] = react_parameter_resolver,
        max_steps: int = 5,
        final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
        streaming: bool = True,
    ):
        """
        Creates a new ConversationalAgent instance

        :param prompt_node: A PromptNode used to communicate with LLM.
        :param prompt_template: A string or PromptTemplate instance to use as the prompt template. If no prompt_template
        is provided, the agent will use the "conversational-agent" template.
        :param tools_manager: A ToolsManager instance to manage tools for the agent. If no tools_manager is provided,
        the agent will install an empty ToolsManager instance.
        :param max_steps: The maximum number of steps for the agent to take, defaults to 5.
        :param memory: A memory object for storing conversation history and other relevant data, defaults to
        ConversationSummaryMemory if no memory is provided.
        :param prompt_parameters_resolver: An optional callable for resolving prompt template parameters,
        defaults to keys: query, tool_names, tool_names_with_descriptions, transcript. Their values are set appropriately.
        :param final_answer_pattern: A regular expression to extract the final answer from the text the Agent generated.
        :param streaming: Whether to use streaming or not. If True, the Agent will stream response tokens from the LLM.
        If False, the Agent will wait for the LLM to finish generating the response and then process it. The default is
        True.
        """

        super().__init__(
            prompt_node=prompt_node,
            prompt_template=prompt_template
            if prompt_template
            else prompt_node.default_prompt_template or "conversational-agent",
            tools_manager=tools_manager,
            memory=memory,
            prompt_parameters_resolver=prompt_parameters_resolver,
            max_steps=max_steps,
            final_answer_pattern=final_answer_pattern,
            streaming=streaming,
        )
