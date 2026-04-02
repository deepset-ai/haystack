"""
Example: Automatic context compaction with SummarizationCompactionTool

This example shows how to add automatic chat history summarization to a Haystack
Agent using the new SummarizationCompactionTool and the condition-triggered tool
mechanism.

The agent has a simple calculator tool and is given a pre-existing conversation
history with alternating user/assistant messages. Once the non-system message count
exceeds `max_messages`, the compaction tool automatically summarizes the oldest
messages into a single concise entry — without the agent having to decide to do it.

Run with:
    OPENAI_API_KEY=<your-key> python compaction_example.py
"""

from typing import Annotated

from haystack.components.agents import Agent
from haystack.components.agents.compaction import SummarizationCompactionTool
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool


@tool
def add(a: Annotated[float, "First number"], b: Annotated[float, "Second number"]) -> float:
    """Add two numbers."""
    return a + b


# A dedicated generator for the compaction summarizer (can be the same model or a
# cheaper/faster one — here we reuse the same model for simplicity).
summarizer_generator = OpenAIChatGenerator(model="gpt-4o-mini")

# The compaction tool fires automatically before each LLM call once the
# non-system message count exceeds max_messages. It summarizes the oldest
# half of the history and keeps the most recent messages verbatim.
compaction = SummarizationCompactionTool(
    chat_generator=summarizer_generator,
    max_messages=6,  # low threshold so it triggers quickly in this demo
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[add, compaction],
    system_prompt="You are a helpful assistant that can perform calculations.",
)

# Pre-existing conversation history with alternating user/assistant messages.
# This simulates a session that has already been running for a while and will
# push the message count past the compaction threshold on the next agent call.
messages = [
    ChatMessage.from_user("What is 3 + 4?"),
    ChatMessage.from_assistant("3 + 4 = 7"),
    ChatMessage.from_user("And 10 + 5?"),
    ChatMessage.from_assistant("10 + 5 = 15"),
    ChatMessage.from_user("What's 8 + 8?"),
    ChatMessage.from_assistant("8 + 8 = 16"),
    ChatMessage.from_user("Can you add 20 and 30?"),
    ChatMessage.from_assistant("20 + 30 = 50"),
    ChatMessage.from_user("Now add 99 and 1."),
]

print(f"History length before agent call: {len(messages)} messages")
print("Compaction threshold: 6 non-system messages — compaction will fire.\n")

result = agent.run(messages=messages)

print(f"History length after agent call:  {len(result['messages'])} messages")
print(f"\nAgent reply: {result['last_message'].text}")
