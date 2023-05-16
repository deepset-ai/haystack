import os

from haystack.agents import Agent, Tool
from haystack.nodes import PromptNode
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

agent_prompt_node = PromptNode(
    "gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=512, stop_words=["Observation:"]
)
agent = Agent(agent_prompt_node)

qa_prompt_node = PromptNode(
    "gpt-3.5-turbo",
    api_key=openai_key,
    max_length=256,
    default_prompt_template="question-answering-with-document-scores",
)
web_retriever = WebRetriever(api_key=search_key)
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=qa_prompt_node)

web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)
agent.add_tool(web_qa_tool)

result = agent.run("What year was the creator of the Python programming language born?")
print(f"\n{result}")
