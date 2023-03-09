import os
import re
from typing import Dict, Any, List

from haystack import Document
from haystack.agents import Agent, Tool
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes.search_engine import WebSearch, NeuralWebSearch

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

ws = WebSearch(api_key=search_key)

pn = PromptNode("text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)

prompt_text = """
Answer the following question using the paragraphs below as sources. An answer should be short, a few words at most.
Provide the answer as the last generated line starting with Answer:

Paragraphs: $paragraphs

Question: $question

Instructions: Consider all the paragraphs above and their corresponding scores to generate the answer. While a single
paragraph may have a high score, it's important to consider all paragraphs for the same answer candidate to answer
accurately.

Let's think step-by-step, we have the following distinct answer possibilities:

"""
pt = PromptTemplate("neural_web_search", prompt_text=prompt_text)


def prepare_prompt_params(results: List[Document], invocation_context: Dict[str, Any]):
    paragraphs = "\n".join([f"-[{doc.meta['score']}] {doc.content}" for doc in results])
    return {"paragraphs": paragraphs, "question": invocation_context.get("query")}


def prepare_final_answer(prompt_node_response: str):
    answer_text = "No answer"
    answer_regex = r"Answer:\s*(.*)"
    answer_match = re.search(answer_regex, prompt_node_response)

    if answer_match:
        answer_text = answer_match.group(1).strip()
    return answer_text


ns = NeuralWebSearch(
    websearch=ws,
    prompt_node=pn,
    prompt_template=pt,
    prepare_template_params_fn=prepare_prompt_params,
    extract_final_answer_fn=prepare_final_answer,
)

questions = [
    "Who won the 1971 San Francisco mayoral election?",
    "Where was Jeremy McKinnon born?",
    "What river is near Dundalk, Ireland?",
    "Who is Kyle Moran?",
    "What party does Joseph Alioto belong to?",
    "When was the Democratic Party founded?",
    "Who is Olivia Wilde's boyfriend?",
]

# for q in questions:
#     response, _ = ns.run(q)
#     print(f"{q} - {response['output']}")

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
Question: $query
Thought:
"""
few_shot_agent_template = PromptTemplate("few-shot-react", prompt_text=few_shot_prompt)
prompt_node = PromptNode(
    "text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"), max_length=512, stop_words=["Observation:"]
)

neural_search_tool = Tool(
    name="Search",
    pipeline_or_node=ns,
    description="useful for when you need to Google questions.",
    output_variable="output",
)

agent = Agent(
    prompt_node=prompt_node,
    prompt_template=few_shot_agent_template,
    tools=[neural_search_tool],
    final_answer_pattern=r"Final Answer\s*:\s*(.*)",
)

hotpot_questions = [
    "What year was the father of the Princes in the Tower born?",
    "Name the movie in which the daughter of Noel Harrison plays Violet Trefusis.",
    "Where was the actress who played the niece in the Priest film born?",
    "Which author is English: John Braine or Studs Terkel?",
]

for q in hotpot_questions:
    result = agent.run(query=q)
    print(result)
