import os

from haystack.agents.base import ConversationalAgentWithTools, Tool, ToolsManager
from haystack.agents.memory import ConversationSummaryMemory
from haystack.agents.types import Color
from haystack.nodes import PromptNode, WebRetriever, PromptTemplate
from haystack.pipelines import WebQAPipeline

few_shot_prompt = """
In the conversation below, a Human interacts with an AI Agent. The Human asks questions, and the AI goes through several steps to answer each question. If the AI Agent is unsure about the answer or if the information might be outdated or inaccurate, it uses the available tools when necessary to find the most up-to-date and accurate information. The AI has access to these tools:
{tool_names_with_descriptions}
Each AI output line starts with a Thought, Tool, Tool Input, Observation, or Final Answer. An Observation marks the beginning of the tool result, and the AI trusts these results.

Examples:
##
Question: What is the capital of France?
Thought: This is a common-knowledge question. Therefore, I'll provide the final answer directly without consulting any tools.
Final Answer: The capital of France is Paris.
##
Question: Who was the first president of the United States?
Thought: This is a common knowledge question. I'll provide the final answer directly without using any tools.
Final Answer: The first president of the United States was George Washington.
##
Question: What is the latest version of Python programming language?
Thought: As of my knowledge cutoff date in September 2021, I'll use the search tool to help me answer the question.
Tool: Search
Tool Input: What is the latest version of Python programming language as of today?
Observation: 3.11.2
Thought: We've learned that the latest Python version is 3.11.2! Now I can give the final answer.
Final Answer: The latest Python version is 3.11.2.
##
Question: What happened with the latest SpaceX Starship launch?
Thought: I can't answer most recent events due to my knowledge cutoff date; I'll use the search tool to help me answer the question.
Tool: Search
Tool Input: What happened with the latest SpaceX Starship launch?
Observation: SpaceX's Starship rocket launched for the first time on Thursday but exploded in mid-flight, falling short of reaching space. No crew were on board. Before the mid-flight failure, the Super Heavy booster successfully separated from the rocket, flipped and began its return to Earth.
Thought: We've learned what happened with the latest SpaceX Starship launch. I can now summarize the final answer.
Final Answer: The latest Space X Starship launch was an experimental launch that did not reach space and ended in mid-flight failure.

Question: What happened after that?
Thought: I'm not sure what you're referring to. I'll use conversation history to help me understand the context.
Tool: conversation_history
Tool Input: What happened after that?
Observation: Human asked AI about the indictment of the former president Trump and AI responded the details of the indictment.
Thought: Based on the conversation history, it seems that the Human is referring events after former President Trump's indictment. I'll use the search tool to help me answer the question.
Tool: Search
Tool Input: What happened after the indictment of the former president Trump?
Observation: The former president Trump went to trial and the trial is still ongoing. I can now give the final answer.
Final Answer: The former president Trump went to trial and the trial is still ongoing.


The AI Agent should use the conversation history tool to infer context when needed, rather than asking for more clarification.
The AI Agent should never respond that a question is too vague. If it is vague, the Agent should consult conversation history tool to better understand the context.

Question: {query}
Thought:
{transcript}
"""
search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not search_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

prompt_text = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than 20 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""

prompt_node = PromptNode(
    "gpt-3.5-turbo",
    default_prompt_template=PromptTemplate("lfqa", prompt_text=prompt_text),
    api_key=openai_key,
    max_length=256,
)

web_retriever = WebRetriever(api_key=search_key, top_search_results=2, mode="preprocessed_documents")
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=prompt_node)

few_shot_agent = PromptTemplate("conversational-agent-with-tools", prompt_text=few_shot_prompt)
conversation_history = Tool(
    name="conversation_history",
    pipeline_or_node=lambda tool_input, **kwargs: agent.memory.load(),
    description="useful for when you need to remember what you've already discussed.",
    logging_color=Color.MAGENTA,
)
web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)
pn = PromptNode(
    "gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.7},
)
agent = ConversationalAgentWithTools(
    pn,
    max_steps=10,
    prompt_template=few_shot_agent,
    tools_manager=ToolsManager(tools=[web_qa_tool, conversation_history]),
    memory=ConversationSummaryMemory(pn),
)

while True:
    user_input = input("\nHuman (type 'exit' or 'quit' to quit): ")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        break
    if user_input.lower() == "memory":
        print("\nMemory:\n", agent.memory.load())
    else:
        assistant_response = agent.run(user_input)
