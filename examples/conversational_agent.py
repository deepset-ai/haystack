import os

from haystack.agents.base import Tool, ToolsManager
from haystack.agents.conversational import ConversationalAgent
from haystack.nodes import PromptNode, WebRetriever, PromptTemplate
from haystack.pipelines import WebQAPipeline
from haystack.agents.types import Color

search_api_key = os.environ.get("SEARCH_API_KEY")
if not search_api_key:
    raise ValueError("Please set the SEARCH_API_KEY environment variable")
model_api_key = os.environ.get("OPENAI_API_KEY")
if not model_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

web_prompt = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise answer, no longer than 10-20 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""

web_prompt_node = PromptNode(
    "gpt-3.5-turbo", default_prompt_template=PromptTemplate(prompt=web_prompt), api_key=model_api_key
)
web_retriever = WebRetriever(api_key=search_api_key, top_search_results=3, mode="snippets")
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=web_prompt_node)
web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)

agent_prompt_node = PromptNode(
    "gpt-3.5-turbo", api_key=model_api_key, stop_words=["Observation:"], model_kwargs={"temperature": 0.5, "top_p": 0.9}
)

conversation_history = Tool(
    name="conversation_history",
    pipeline_or_node=lambda tool_input, **kwargs: agent.memory.load(),  # type: ignore
    description="useful for when you need to remember what you've already discussed.",
    logging_color=Color.MAGENTA,
)

agent = ConversationalAgent(
    prompt_node=agent_prompt_node, max_steps=6, tools_manager=ToolsManager(tools=[web_qa_tool, conversation_history])
)

test = False
if test:
    questions = [
        "Why was Jamie Foxx recently hospitalized?",
        "Where was he hospitalized?",
        "What movie was he filming at the time?",
        "Who is Jamie's female co-star in the movie he was filing at that time?",
        "Tell me more about her, who is her partner?",
    ]
    for question in questions:
        agent.run(question)
else:
    while True:
        user_input = input("\nHuman (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        assistant_response = agent.run(user_input)
