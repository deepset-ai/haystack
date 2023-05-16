from typing import Optional, Callable

from haystack.agents import Agent
from haystack.agents.memory import Memory, ConversationMemory
from haystack.nodes import PromptNode


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
        prompt_parameters_resolver: Optional[Callable] = None,
    ):
        """
        Creates a new ConversationalAgent instance

        :param prompt_node: A PromptNode used to communicate with LLM.
        :param memory: A memory instance for storing conversation history and other relevant data, defaults to
        ConversationMemory.
        :param prompt_parameters_resolver: An optional callable for resolving prompt template parameters,
        defaults to a callable that returns a dictionary with the query and the conversation history.
        """
        super().__init__(
            prompt_node=prompt_node,
            prompt_template=prompt_node.default_prompt_template
            if prompt_node.default_prompt_template is not None
            else "conversational-agent",
            max_steps=2,
            memory=memory if memory else ConversationMemory(),
            prompt_parameters_resolver=prompt_parameters_resolver
            if prompt_parameters_resolver
            else lambda query, agent, **kwargs: {"query": query, "history": agent.memory.load()},
            final_answer_pattern=r"^([\s\S]+)$",
        )
