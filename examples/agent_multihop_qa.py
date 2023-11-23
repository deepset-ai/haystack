import os

from haystack.agents import Agent, Tool
from haystack.agents.base import ToolsManager
from haystack.agents.types import AgentToolLogger
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


pn = PromptNode(
    "gpt-3.5-turbo",
    api_key=openai_key,
    max_length=256,
    default_prompt_template="question-answering-with-document-scores",
)
web_retriever = WebRetriever(api_key=search_key)
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)

few_shot_prompt = """
You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions correctly, you have access to the following tools:

Search: useful for when you need to Google questions. You should ask targeted questions, for example, Who is Anthony Dirrell's brother?

To answer questions, you'll need to go through multiple steps involving step-by-step thinking and selecting appropriate tools and their inputs; tools will respond with observations. When you are ready for a final answer, respond with the `Final Answer:`
Examples:
##
Question: Anthony Dirrell is the brother of which super middleweight title holder?
Thought: Let's think step by step. To answer this question, we first need to know who Anthony Dirrell is.
Tool: Search
Tool Input: Who is Anthony Dirrell?
Observation: Boxer
Thought: We've learned Anthony Dirrell is a Boxer. Now, we need to find out who his brother is.
Tool: Search
Tool Input: Who is Anthony Dirrell brother?
Observation: Andre Dirrell
Thought: We've learned Andre Dirrell is Anthony Dirrell's brother. Now, we need to find out what title Andre Dirrell holds.
Tool: Search
Tool Input: What is the Andre Dirrell title?
Observation: super middleweight
Thought: We've learned Andre Dirrell title is super middleweight. Now, we can answer the question.
Final Answer: Andre Dirrell
##
Question: What year was the party of the winner of the 1971 San Francisco mayoral election founded?
Thought: Let's think step by step. To answer this question, we first need to know who won the 1971 San Francisco mayoral election.
Tool: Search
Tool Input: Who won the 1971 San Francisco mayoral election?
Observation: Joseph Alioto
Thought: We've learned Joseph Alioto won the 1971 San Francisco mayoral election. Now, we need to find out what party he belongs to.
Tool: Search
Tool Input: What party does Joseph Alioto belong to?
Observation: Democratic Party
Thought: We've learned Democratic Party is the party of Joseph Alioto. Now, we need to find out when the Democratic Party was founded.
Tool: Search
Tool Input: When was the Democratic Party founded?
Observation: 1828
Thought: We've learned the Democratic Party was founded in 1828. Now, we can answer the question.
Final Answer: 1828
##
Question: Right Back At It Again contains lyrics co-written by the singer born in what city?
Thought: Let's think step by step. To answer this question, we first need to know what song the question is referring to.
Tool: Search
Tool Input: What is the song Right Back At It Again?
Observation: "Right Back at It Again" is the song by A Day to Remember
Thought: We've learned Right Back At It Again is a song by A Day to Remember. Now, we need to find out who co-wrote the song.
Tool: Search
Tool Input: Who co-wrote the song Right Back At It Again?
Observation: Jeremy McKinnon
Thought: We've learned Jeremy McKinnon co-wrote the song Right Back At It Again. Now, we need to find out what city he was born in.
Tool: Search
Tool Input: Where was Jeremy McKinnon born?
Observation: Gainsville, Florida
Thought: We've learned Gainsville, Florida is the city Jeremy McKinnon was born in. Now, we can answer the question.
Final Answer: Gainsville, Florida
##
Question: {query}
Thought:
{transcript}
"""
few_shot_agent_template = PromptTemplate(few_shot_prompt)
prompt_node = PromptNode(
    "gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=512, stop_words=["Observation:"]
)

web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)

agent = Agent(
    prompt_node=prompt_node, prompt_template=few_shot_agent_template, tools_manager=ToolsManager([web_qa_tool])
)
atl = AgentToolLogger(agent_events=agent.callback_manager, tool_events=agent.tm.callback_manager)

hotpot_questions = [
    "What year was the father of the Princes in the Tower born?",
    "Name the movie in which the daughter of Noel Harrison plays Violet Trefusis.",
    "Where was the actress who played the niece in the Priest film born?",
    "Which author is English: John Braine or Studs Terkel?",
]
verbose = False
for question in hotpot_questions:
    result = agent.run(query=question)
    print(f"\n{result}")
    if verbose:
        print(f"\n{atl.logs}")
