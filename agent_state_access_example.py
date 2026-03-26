# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
POC: Agent State Access - skills-as-state pattern

The agent has two skills stored as strings in state:
  - summarize_skill: a prompt telling the agent how to summarise text
  - translate_skill: a prompt telling the agent how to translate text

Rather than loading both into the system prompt upfront, the agent discovers
available skills via ls_state, loads the one it needs via read_state, and writes
its final output to a `final_answer` key via write_state.
"""

from haystack.components.agents import Agent
from haystack.components.agents.state import LsStateTool, ReadStateTool, WriteStateTool
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-5.4"),
    tools=[LsStateTool(), ReadStateTool(), WriteStateTool()],
    state_schema={"summarize_skill": {"type": str}, "translate_skill": {"type": str}, "final_answer": {"type": str}},
    system_prompt="""You are a helpful assistant.
Use ls_state to discover available state keys, read_state to read their values, and write_state to record your final
response in the `final_answer` key.

If you see a key that ends in `_skill`, it contains instructions for how to perform a specific task.
Use these instructions to guide your actions.""",
)

result = agent.run(
    messages=[
        ChatMessage.from_user(
            """Please summarise the following text:

Haystack is an open-source AI orchestration framework that you can use to build powerful, production-ready applications with Large Language Models (LLMs) for various use cases. Whether you’re creating autonomous agents, multimodal apps, or scalable RAG systems, Haystack provides the tools to move from idea to production easily.
Haystack is designed in a modular way, allowing you to combine the best technology from OpenAI, Google, Anthropic, and open-source projects like Hugging Face's Transformers.
The core foundation of Haystack consists of components and pipelines, along with Document Stores, Agents, Tools, and many integrations. Read more about Haystack concepts in the Haystack Concepts Overview.
Supported by an engaged community of developers, Haystack has grown into a comprehensive and user-friendly framework for LLM-based development.
"""  # noqa: E501
        )
    ],
    summarize_skill=(
        "To summarise text: identify the main topic, strip filler words, and return a single concise sentence."
    ),
    translate_skill=(
        "To translate text: preserve meaning and tone exactly, and return only the translated text without commentary."
    ),
    streaming_callback=print_streaming_chunk,
)

print("Final answer:", result["final_answer"])
