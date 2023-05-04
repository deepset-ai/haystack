from typing import Union, Optional, Callable

from haystack.agents import Agent
from haystack.agents.answer_parser import AgentAnswerParser, BasicAnswerParser
from haystack.agents.base import ToolsManager, CallablePromptParametersResolver
from haystack.agents.memory import Memory, ConversationSummaryMemory
from haystack.agents.types import PromptParametersResolver
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
        memory: Optional[Memory] = None,
        prompt_parameters_resolver: Optional[Union[PromptParametersResolver, Callable]] = None,
    ):
        """
        Creates ConversationalAgent

        :param prompt_node: A PromptNode used to communicate with LLM.
        :param memory: A memory object for storing conversation history and other relevant data, defaults to None.
        :param prompt_parameters_resolver: An optional resolver or callable for resolving prompt template parameters,
        defaults to None.
        """
        super().__init__(
            prompt_node=prompt_node,
            prompt_template=prompt_node.default_prompt_template
            if prompt_node.default_prompt_template is not None
            else "conversational-agent",
            max_steps=2,
            memory=memory if memory else ConversationSummaryMemory(prompt_node),
            prompt_parameters_resolver=prompt_parameters_resolver
            if prompt_parameters_resolver
            else CallablePromptParametersResolver(
                lambda query, agent, **kwargs: {"query": query, "history": agent.memory.load(keys=["history"])}
            ),
            final_answer_pattern=BasicAnswerParser(),
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
        prompt_parameters_resolver: Optional[Union[PromptParametersResolver, Callable]] = None,
        final_answer_pattern: Union[str, AgentAnswerParser] = r"Final Answer\s*:\s*(.*)",
    ):
        """
        Creates a ConversationalAgentWithTools.

        :param prompt_node: A PromptNode used to communicate with LLM.
        :param prompt_template: A string or PromptTemplate instance for the prompt, defaults to None.
        :param tools_manager: A ToolsManager instance to manage tools for the agent, defaults to None.
        :param max_steps: The maximum number of steps for the agent to take, defaults to 5.
        :param memory: A memory object for storing conversation history and other relevant data, defaults to None.
        :param prompt_parameters_resolver: An optional resolver or callable for resolving prompt template parameters,
        defaults to None.
        :param final_answer_pattern: A string or AgentAnswerParser instance for parsing the final answer, defaults to
        a regex pattern capturing "Final Answer: " followed by any text.
        """
        super().__init__(
            prompt_node=prompt_node,
            prompt_template=prompt_template,
            tools_manager=tools_manager,
            max_steps=max_steps,
            memory=memory if memory else ConversationSummaryMemory(prompt_node),
            prompt_parameters_resolver=prompt_parameters_resolver
            if prompt_parameters_resolver
            else CallablePromptParametersResolver(
                lambda query, agent, agent_step, **kwargs: {
                    "query": query,
                    "tool_names": agent.tm.get_tool_names(),
                    "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
                    "transcript": agent_step.transcript,
                    "history": agent.memory.load(keys=["history"]),
                }
            ),
            final_answer_pattern=final_answer_pattern,
        )
