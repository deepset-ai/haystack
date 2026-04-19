# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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


# A dedicated generator for the compaction summarizer (can be the same model or a cheaper/faster one — here we reuse
# the same model for simplicity).
summarizer_generator = OpenAIChatGenerator(model="gpt-4o-mini")

# The compaction tool fires automatically before each LLM call once the non-system message count exceeds max_messages.
# It summarizes the oldest half of the history and keeps the most recent messages verbatim.
compaction = SummarizationCompactionTool(
    chat_generator=summarizer_generator,
    max_messages=3,  # low threshold so it triggers quickly in this demo
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
    ChatMessage.from_user("What is quantum mechanics?"),
    ChatMessage.from_assistant(
        "Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at "
        "the smallest scales."
    ),
    ChatMessage.from_user("Can you explain it in simple terms?"),
    ChatMessage.from_assistant(
        "Sure! Quantum mechanics is like a set of rules that govern how tiny particles, like electrons and photons,"
        " behave. It tells us that these particles can exist in multiple states at once (superposition) and can be "
        "connected in ways that seem to defy classical physics (entanglement). It's a fascinating and complex field "
        "that has led to many technological advancements, like semiconductors and quantum computing."
    ),
    ChatMessage.from_user("Thanks! Now, can you do some math for me? What's 3 + 4?"),
]

print(f"History length before agent call: {len(messages)} messages")

result = agent.run(messages=messages)

print(f"History length after agent call: {len(result['messages'])} messages")
print(f"\nAgent reply: {result['last_message'].text}")
