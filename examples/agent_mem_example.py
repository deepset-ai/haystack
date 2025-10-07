import os
from typing import List

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.memory import Mem0MemoryStore
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from haystack.tools.tool import Tool


@tool
def save_user_preference(preference_type: str, preference_value: str) -> str:
    """Save user preferences that should be remembered"""
    return f"✅ Saved preference: {preference_type} = {preference_value}"


@tool
def get_recommendation(category: str) -> str:
    """Get personalized recommendations based on user preferences"""
    recommendations = {
        "food": "Based on your preferences, try the Mediterranean cuisine!",
        "music": "I recommend some jazz playlists for you!",
        "books": "You might enjoy science fiction novels!",
    }
    return recommendations.get(category, "I'll learn your preferences to give better recommendations!")


memory_store = Mem0MemoryStore(api_key=os.getenv("MEM0_API_KEY"))
memory_store.set_memory_config(user_id="test_123")

"""
User can setup opensearch config and filters using memory_store.set_memory_config() e.g.

memory_store.set_memory_config( user_id="test_123",
                                backend_config=opensearch_config,
                                filters={"categories": {"contains": "movie"}})
"""


# Agent Setup
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    memory_store=memory_store,
    tools=[save_user_preference, get_recommendation],
    system_prompt="""
    You are a personal assistant with memory capabilities.
    Use the provided memories to personalize your responses and remember user context.
    When users share preferences, use the save_user_preference tool.
    When asked for recommendations, use the get_recommendation tool.
    Be conversational and reference previous interactions when relevant.
    """,
    exit_conditions=["text"],
    state_schema={"text": {"type": str}},
)

# Run the Agent
agent.warm_up()
response = agent.run(messages=[ChatMessage.from_user("Recommend me a movie to watch on Friday night.")])
