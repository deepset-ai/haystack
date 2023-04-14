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
